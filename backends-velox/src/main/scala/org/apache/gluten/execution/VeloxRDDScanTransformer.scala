/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.gluten.execution

import org.apache.gluten.backendsapi.velox.VeloxValidatorApi
import org.apache.gluten.config.{GlutenConfig, VeloxConfig}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{RDDScanTransformer, SparkPlan}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Velox-backend implementation of RDDScanTransformer.
 *
 * Converts an RDD[InternalRow] into columnar batches using Velox's native row-to-columnar
 * conversion (same JNI path as RowToVeloxColumnarExec).
 */
case class VeloxRDDScanTransformer(
    outputAttributes: Seq[Attribute],
    rdd: RDD[InternalRow],
    name: String,
    // Row-to-columnar conversion preserves data distribution, so we carry through
    // the original partitioning. This differs from CH which uses UnknownPartitioning(0)
    // but is consistent with RowToVeloxColumnarExec's behavior.
    override val outputPartitioning: Partitioning,
    override val outputOrdering: Seq[SortOrder]
) extends RDDScanTransformer(outputAttributes, outputPartitioning, outputOrdering)
  with Logging {

  override def nodeName: String = name

  @transient override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputRows" -> SQLMetrics.createMetric(sparkContext, "number of input rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "convertTime" -> SQLMetrics.createTimingMetric(sparkContext, "time to convert")
  )

  override protected def doValidateInternal(): ValidationResult = {
    for (field <- schema.fields) {
      val reason = VeloxValidatorApi.validateSchema(field.dataType)
      if (reason.isDefined) {
        return ValidationResult.failed(reason.get)
      }
      val arrowReason = validateArrowCompatibility(field.dataType)
      if (arrowReason.isDefined) {
        return ValidationResult.failed(arrowReason.get)
      }
    }

    ValidationResult.succeeded
  }

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numInputRows = longMetric("numInputRows")
    val numOutputBatches = longMetric("numOutputBatches")
    val convertTime = longMetric("convertTime")
    val localSchema = this.schema
    val batchSize = GlutenConfig.get.maxBatchSize
    val batchBytes = VeloxConfig.get.veloxPreferredBatchBytes
    rdd.mapPartitions {
      iter =>
        if (iter.hasNext) {
          val first = iter.next()
          first match {
            case _: BatchCarrierRow =>
              // RDD already contains columnar batches wrapped as carrier rows
              // (e.g., from df.checkpoint() on a Gluten plan). Unwrap directly.
              (Iterator.single(first) ++ iter).flatMap {
                row =>
                  BatchCarrierRow.unwrap(row).map {
                    batch =>
                      numOutputBatches += 1
                      numInputRows += batch.numRows()
                      batch
                  }
              }
            case _ =>
              // Standard InternalRow path - convert via native row-to-columnar.
              RowToVeloxColumnarExec.toColumnarBatchIterator(
                Iterator.single(first) ++ iter,
                localSchema,
                numInputRows,
                numOutputBatches,
                convertTime,
                batchSize,
                batchBytes)
          }
        } else {
          Iterator.empty
        }
    }
  }

  /**
   * Additional validation for Arrow export compatibility. The RDDScan path transfers data via Arrow
   * ABI, which has stricter constraints than Velox's type system:
   *   - Map types can trigger "Map data key type should be a non-nullable" in Arrow export
   *   - Interval types are not supported by ArrowWritableColumnVector
   */
  private def validateArrowCompatibility(dataType: DataType): Option[String] = {
    dataType match {
      case _: MapType =>
        Some(s"Map type is not supported in RDDScan Arrow export path: $dataType")
      case _: YearMonthIntervalType | _: DayTimeIntervalType | CalendarIntervalType =>
        Some(s"Interval type is not supported in Arrow export: $dataType")
      case struct: StructType =>
        struct.fields.flatMap(f => validateArrowCompatibility(f.dataType)).headOption
      case array: ArrayType =>
        validateArrowCompatibility(array.elementType)
      case _ => None
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan = {
    assert(newChildren.isEmpty, "VeloxRDDScanTransformer is a leaf node")
    copy(outputAttributes, rdd, name, outputPartitioning, outputOrdering)
  }
}

object VeloxRDDScanTransformer {
  def replace(plan: org.apache.spark.sql.execution.RDDScanExec): RDDScanTransformer =
    VeloxRDDScanTransformer(
      plan.output,
      plan.inputRDD,
      plan.nodeName,
      plan.outputPartitioning,
      plan.outputOrdering)
}
