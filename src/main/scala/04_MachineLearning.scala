import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, StandardScaler}
import java.io.{FileWriter, BufferedWriter}

object MachineLearning {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("LinkedIn Jobs Machine Learning - Phase 5")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val filePath   = "src/main/resources/transformed_linkedin_jobs.csv"
    val outputPath = "ML_output.txt"
    val fw = new BufferedWriter(new FileWriter(outputPath))

    def log(line: String = ""): Unit = { println(line); fw.write(line + "\n") }
    def sep(): Unit = log("=" * 70)

    // ============================================================
    // 1 — FEATURE ENGINEERING & DATA PREPARATION 
    // ============================================================

    sep()
    log(" 1: Feature Engineering & Data Preparation")
    sep()

    // ─────────────────────────────────────────────────────────────
    // STEP 1: LOAD & CLEAN DATASET
    // Filtering "Original" records and casting to Double to ensure
    // compatibility with the Spark ML stages (Encoding & Scaling).
    // ─────────────────────────────────────────────────────────────
    log("\n----- STEP 1: LOAD & CLEAN DATASET -----")

    val dfRaw = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)
    
    val dfFiltered = dfRaw.filter(col("salary_source") === "Original")
      .withColumn("normalized_salary", col("normalized_salary").cast("double"))
      .withColumn("state_idx_num", col("state_idx_num").cast("double"))
      .withColumn("formatted_work_type_idx_num", col("formatted_work_type_idx_num").cast("double"))
      .withColumn("title_idx_num", col("title_idx_num").cast("double"))
      .withColumn("formatted_experience_level_idx_num", col("formatted_experience_level_idx_num").cast("double"))
      .withColumn("remote_allowed", col("remote_allowed").cast("double"))

    log(s"  Total rows loaded  : ${dfRaw.count()}")
    log(s"  Rows after filter  : ${dfFiltered.count()}")

    // ─────────────────────────────────────────────────────────────
    // STEP 2: TRAIN / TEST SPLIT (80/20)
    // Splitting occurs before any feature transformations to prevent 
    // Data Leakage from the test set into the training process.
    // ─────────────────────────────────────────────────────────────
    log("\n----- STEP 2: TRAIN / TEST SPLIT (80/20) -----")

    val Array(trainData, testData) = dfFiltered.randomSplit(Array(0.8, 0.2), seed = 42)

    log(s"  Training rows      : ${trainData.count()}")
    log(s"  Test rows          : ${testData.count()}")

    // ─────────────────────────────────────────────────────────────
    // STEP 3: PIPELINE CONSTRUCTION (Feature Engineering)
    // Constructing transformation stages in a logical order.
    // Note: Model Training will be added as the final stage.
    // ─────────────────────────────────────────────────────────────
    log("\n----- STEP 3: PIPELINE CONSTRUCTION -----")

    // Stage A: One-Hot Encoding for categorical features
    val encoder = new OneHotEncoder()
      .setInputCols(Array("state_idx_num", "formatted_work_type_idx_num", "title_idx_num"))
      .setOutputCols(Array("stateVec", "workTypeVec", "titleVec"))

    // Stage B: Vector Assembler to combine all features
    val featureCols = Array("stateVec", "workTypeVec", "titleVec", "formatted_experience_level_idx_num", "remote_allowed")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features_raw")

    // Stage C: Standard Scaling for normalization
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithMean(true)
      .setWithStd(true)

    // Combine stages into a partial pipeline (to be completed by Person 2)
    val pipelineStages = Array(encoder, assembler, scaler)
    val pipeline = new Pipeline().setStages(pipelineStages)

    log("  Pipeline Stages Ready: Encoder -> Assembler -> Scaler")

    sep()
    log(" 1 COMPLETE")
    sep()

    fw.close()
    spark.stop()
  }
}