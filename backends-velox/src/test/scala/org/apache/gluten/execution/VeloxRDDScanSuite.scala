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
package org.apache.spark.sql.execution

import org.apache.gluten.execution._

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.classic.{ClassicDataset, ClassicTypes}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.types._
import org.apache.spark.util.Utils

class VeloxRDDScanSuite extends VeloxWholeStageTransformerSuite with AdaptiveSparkPlanHelper {

  override protected val resourcePath: String = "/tpch-data-parquet"
  override protected val fileFormat: String = "parquet"

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      .set("spark.sql.ansi.enabled", "false")
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
    createTPCHNotNullTables()
  }

  /** Creates a DataFrame backed by LogicalRDD/RDDScanExec from an existing DataFrame. */
  private def asRDDScanDF(data: DataFrame): DataFrame = {
    val node = LogicalRDD(data.queryExecution.analyzed.output, data.queryExecution.toRdd)(
      data.sparkSession.asInstanceOf[ClassicTypes.ClassicSparkSession])
    ClassicDataset.ofRows(spark, node).toDF()
  }

  test("basic RDDScanExec is replaced by VeloxRDDScanTransformer") {
    val data = spark.sql("SELECT l_orderkey, l_partkey FROM lineitem LIMIT 10")
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with string and numeric types") {
    val data = spark.sql("""SELECT l_returnflag, l_linestatus, l_quantity, l_extendedprice
                           |FROM lineitem LIMIT 20""".stripMargin)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with aggregation downstream") {
    val query =
      """SELECT l_returnflag, sum(l_quantity) AS sum_qty
        |FROM lineitem
        |WHERE l_shipdate <= date'1998-09-02'
        |GROUP BY l_returnflag""".stripMargin
    val data = spark.sql(query)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with empty RDD") {
    val data = spark.sql("SELECT l_orderkey FROM lineitem WHERE 1 = 0")
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    assert(df.count() == 0)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan preserves data correctness with multiple re-reads") {
    val data = spark.sql("SELECT l_orderkey, l_partkey FROM lineitem LIMIT 50")
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    // Read twice to verify idempotency
    checkAnswer(df, expectedAnswer)
    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with null values") {
    val rdd = spark.sparkContext.parallelize(
      Seq(
        Row(1, "a", null),
        Row(null, "b", 2.0),
        Row(3, null, 3.0)
      ))
    val schema = StructType(
      Seq(
        StructField("id", IntegerType, nullable = true),
        StructField("name", StringType, nullable = true),
        StructField("value", DoubleType, nullable = true)
      ))
    val data = spark.createDataFrame(rdd, schema)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with all supported primitive types") {
    val rdd = spark.sparkContext.parallelize(
      Seq(
        Row(
          true,
          1.toByte,
          2.toShort,
          3,
          4L,
          5.0f,
          6.0,
          "hello",
          java.sql.Date.valueOf("2024-01-01"),
          java.sql.Timestamp.valueOf("2024-01-01 12:00:00"),
          Array[Byte](1, 2, 3),
          BigDecimal("123.45").underlying()
        )
      ))
    val schema = StructType(
      Seq(
        StructField("bool", BooleanType),
        StructField("byte", ByteType),
        StructField("short", ShortType),
        StructField("int", IntegerType),
        StructField("long", LongType),
        StructField("float", FloatType),
        StructField("double", DoubleType),
        StructField("string", StringType),
        StructField("date", DateType),
        StructField("timestamp", TimestampType),
        StructField("binary", BinaryType),
        StructField("decimal", DecimalType(10, 2))
      ))
    val data = spark.createDataFrame(rdd, schema)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with array type") {
    val rdd = spark.sparkContext.parallelize(
      Seq(
        Row(Seq(1, 2, 3)),
        Row(Seq(4, 5))
      ))
    val schema = StructType(Seq(StructField("arr", ArrayType(IntegerType))))
    val data = spark.createDataFrame(rdd, schema)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan with map type falls back to row-based") {
    val rdd = spark.sparkContext.parallelize(
      Seq(
        Row(Map("a" -> 1, "b" -> 2)),
        Row(Map("c" -> 3))
      ))
    val schema = StructType(Seq(StructField("m", MapType(StringType, IntegerType))))
    val data = spark.createDataFrame(rdd, schema)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    // MapType is not supported in Arrow export, so falls back to row-based processing
    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.isEmpty, "MapType schema should fall back from VeloxRDDScanTransformer")
  }

  test("RDDScan with struct type") {
    val rdd = spark.sparkContext.parallelize(
      Seq(
        Row(Row("hello", 1)),
        Row(Row("world", 2))
      ))
    val innerSchema =
      StructType(Seq(StructField("name", StringType), StructField("value", IntegerType)))
    val schema = StructType(Seq(StructField("s", innerSchema)))
    val data = spark.createDataFrame(rdd, schema)
    val expectedAnswer = data.collect()
    val df = asRDDScanDF(data)

    checkAnswer(df, expectedAnswer)
    val cnt = collect(df.queryExecution.executedPlan) { case _: VeloxRDDScanTransformer => true }
    assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
  }

  test("RDDScan falls back for unsupported types") {
    val data = spark.sql("SELECT INTERVAL '1' DAY AS di")
    val expectedAnswer = data.collect()
    val result = asRDDScanDF(data)

    // Should still produce correct results via fallback to vanilla Spark
    checkAnswer(result, expectedAnswer)
    val cnt = collect(result.queryExecution.executedPlan) {
      case _: VeloxRDDScanTransformer => true
    }
    assert(cnt.isEmpty, "Expected fallback - VeloxRDDScanTransformer should NOT be in plan")
  }

  test("RDDScan handles BatchCarrierRow from checkpoint") {
    val tempDir = Utils.createTempDir()
    try {
      spark.sparkContext.setCheckpointDir(tempDir.getAbsolutePath)
      val df = spark.range(100).selectExpr("id", "id * 2 as value")
      val checkpointed = df.localCheckpoint()
      val result = asRDDScanDF(checkpointed)

      checkAnswer(result, df.collect())
      val cnt = collect(result.queryExecution.executedPlan) {
        case _: VeloxRDDScanTransformer => true
      }
      assert(cnt.nonEmpty, "Expected VeloxRDDScanTransformer in plan")
    } finally {
      Utils.deleteRecursively(tempDir)
    }
  }

  test("falls back for schemas with interval types") {
    val df = spark.sql("SELECT INTERVAL '1' YEAR as y")
    val rddDf = asRDDScanDF(df)
    checkAnswer(rddDf, df.collect())
    // Should NOT use VeloxRDDScanTransformer (falls back due to interval type)
    val veloxScans = collect(rddDf.queryExecution.executedPlan) {
      case _: VeloxRDDScanTransformer => true
    }
    assert(veloxScans.isEmpty, "Interval type schema should fall back from VeloxRDDScanTransformer")
  }
}
