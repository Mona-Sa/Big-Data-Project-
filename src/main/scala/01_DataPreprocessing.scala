import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import java.io.{FileWriter, BufferedWriter, File}
import java.nio.file.{Files, Paths, StandardCopyOption}

object FullPipeline {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("LinkedIn Jobs Full Pipeline")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    val filePath   = "src/main/resources/linkedin_jobs.csv"
    val outputPath = "full_pipeline_output.txt"
    val csvTempDir = "src/main/resources/cleaned_linkedin_jobs_temp"
    val csvFinal   = "src/main/resources/cleaned_linkedin_jobs.csv"

    val fw = new BufferedWriter(new FileWriter(outputPath))
    def log(line: String = ""): Unit = { println(line); fw.write(line + "\n") }
    def sep(title: String): Unit = {
      log()
      log("=" * 60)
      log(s"  $title")
      log("=" * 60)
    }

    // ══════════════════════════════════════════════════════════════
    // PART 1 — EDA (RAW DATA)
    // ══════════════════════════════════════════════════════════════
    sep("PART 1: EXPLORATORY DATA ANALYSIS (RAW DATA)")

    val dfRaw = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")
      .option("escape", "\"")
      .csv(filePath)
      .withColumn("listed_time_ts",          to_timestamp(from_unixtime(col("listed_time") / 1000)))
      .withColumn("original_listed_time_ts", to_timestamp(from_unixtime(col("original_listed_time") / 1000)))
      .withColumn("expiry_ts",               to_timestamp(from_unixtime(col("expiry") / 1000)))

    var rawCount = dfRaw.count()

    // ── (0) Basic Overview ─────────────────────────────────────────
    log("\n----- (0) BASIC OVERVIEW -----")
    log(s"Rows    : $rawCount")
    log(s"Columns : ${dfRaw.columns.length}")
    log("Column names:")
    log(dfRaw.columns.sorted.mkString(", "))

    // ── (1) Schema ────────────────────────────────────────────────
    log("\n----- (1) SCHEMA -----")
    dfRaw.dtypes.foreach { case (name, dtype) =>
      log(f"  $name%-35s $dtype")
    }

    // ── (2) Missing Values ────────────────────────────────────────
    log("\n----- (2) MISSING VALUES PER COLUMN (RAW) -----")
    dfRaw.columns.foreach { c =>
      val missing = dfRaw.filter(col(c).isNull || (col(c).cast("string") === "")).count()
      val pct = (missing.toDouble / rawCount) * 100.0
      log(f"  $c%-35s missing=${missing}%8d (${pct}%.2f%%)")
    }

    // ── (3) Duplicates ────────────────────────────────────────────
    log("\n----- (3) DUPLICATES CHECK -----")
    val distinctIds = dfRaw.select("job_id").distinct().count()
    log(s"  Distinct job_id     : $distinctIds")
    log(s"  Duplicate job_id    : ${rawCount - distinctIds}")

    // ── (4) Categorical Distributions ────────────────────────────
    val catCols = Seq("formatted_work_type", "formatted_experience_level",
                      "pay_period", "currency", "application_type", "work_type")
    catCols.foreach { c =>
      log(s"\n----- (4) TOP VALUES: $c -----")
      dfRaw.groupBy(c).count().orderBy(desc("count")).limit(10)
        .collect().foreach(r => log(f"  ${String.valueOf(r.get(0))}%-30s ${r.getLong(1)}%,d"))
    }

    // ── (5) Top Locations ─────────────────────────────────────────
    log("\n----- (5) TOP 20 LOCATIONS -----")
    dfRaw.groupBy("location").count().orderBy(desc("count")).limit(20)
      .collect().foreach(r => log(f"  ${r.getString(0)}%-35s ${r.getLong(1)}%,d"))

    // ── (6) Top Companies ─────────────────────────────────────────
    log("\n----- (6) TOP 20 COMPANIES -----")
    dfRaw.groupBy("company_name").count().orderBy(desc("count")).limit(20)
      .collect().foreach(r => log(f"  ${String.valueOf(r.get(0))}%-40s ${r.getLong(1)}%,d"))

    // ── (7) Top Job Titles ────────────────────────────────────────
    log("\n----- (7) TOP 20 JOB TITLES -----")
    dfRaw.groupBy("title").count().orderBy(desc("count")).limit(20)
      .collect().foreach(r => log(f"  ${r.getString(0)}%-40s ${r.getLong(1)}%,d"))

    // ── (8) Numeric Summary ───────────────────────────────────────
    log("\n----- (8) NUMERIC SUMMARY -----")
    val numCols = Seq("min_salary", "max_salary", "normalized_salary", "views", "applies")
    numCols.foreach { c =>
      val stats = dfRaw.select(
        count(col(c)).alias("count"),
        round(avg(col(c)), 2).alias("mean"),
        round(stddev(col(c)), 2).alias("stddev"),
        round(min(col(c)), 2).alias("min"),
        round(max(col(c)), 2).alias("max")
      ).collect()(0)
      log(f"  $c%-20s count=${stats.get(0)}  mean=${stats.get(1)}  stddev=${stats.get(2)}  min=${stats.get(3)}  max=${stats.get(4)}")
    }

    // ── (9) Salary Sanity Check ───────────────────────────────────
    log("\n----- (9) SALARY SANITY CHECK -----")
    val badSalary = dfRaw.filter(col("min_salary") > col("max_salary")).count()
    log(s"  Rows where min_salary > max_salary: $badSalary")

    // ── (10) Posts by Month ───────────────────────────────────────
    log("\n----- (10) POSTS BY MONTH -----")
    dfRaw.withColumn("listed_month", date_format(col("listed_time_ts"), "yyyy-MM"))
      .groupBy("listed_month").count().orderBy("listed_month")
      .collect().foreach(r => log(f"  ${String.valueOf(r.get(0))}%-12s ${r.getLong(1)}%,d"))

    // ── (11) Remote Distribution ──────────────────────────────────
    log("\n----- (11) REMOTE ALLOWED DISTRIBUTION -----")
    dfRaw.groupBy("remote_allowed").count().orderBy(desc("count"))
      .collect().foreach(r => log(f"  ${String.valueOf(r.get(0))}%-10s ${r.getLong(1)}%,d"))

    // ── (12) Salary by Work Type ──────────────────────────────────
    log("\n----- (12) AVG SALARY BY WORK TYPE -----")
    dfRaw.filter(col("normalized_salary").isNotNull)
      .groupBy("formatted_work_type")
      .agg(
        count("*").alias("rows"),
        round(avg("normalized_salary"), 2).alias("avg_salary"),
        round(max("normalized_salary"), 2).alias("max_salary")
      )
      .orderBy(desc("rows"))
      .collect()
      .foreach(r => log(f"  ${String.valueOf(r.get(0))}%-15s rows=${r.getLong(1)}%6d  avg=${r.get(2)}  max=${r.get(3)}"))

    // ══════════════════════════════════════════════════════════════
    // PART 2 — PREPROCESSING PIPELINE
    // ══════════════════════════════════════════════════════════════
    sep("PART 2: PREPROCESSING PIPELINE")

    // Step 1: Drop high-null columns (>95% missing)
    log("\n----- STEP 1: DROP HIGH-NULL COLUMNS -----")
    log("  Dropping: closed_time (99.13%), skills_desc (98.03%), med_salary (94.93%)")
    val dfStep1 = dfRaw.drop("closed_time", "skills_desc", "med_salary")
    log(s"  Columns remaining: ${dfStep1.columns.length}")

    // Step 2: Drop rows missing critical fields
    log("\n----- STEP 2: DROP ROWS WITH NULL job_id OR title -----")
    val dfStep2 = dfStep1.filter(col("job_id").isNotNull && col("title").isNotNull)
    log(s"  Rows after: ${dfStep2.count()}")

    // Step 3: Fill missing categorical values with "Unknown"
    log("\n----- STEP 3: FILL MISSING CATEGORICAL VALUES -----")
    val dfStep3 = dfStep2
      .na.fill("Unknown", Seq(
        "formatted_experience_level",
        "formatted_work_type",
        "company_name",
        "pay_period",
        "currency",
        "compensation_type",
        "application_url",
        "posting_domain"
      ))
    log("  Filled: formatted_experience_level, formatted_work_type, company_name,")
    log("          pay_period, currency, compensation_type, application_url, posting_domain → 'Unknown'")

    // Step 4: Fill missing numeric values with 0
    log("\n----- STEP 4: FILL MISSING NUMERIC VALUES -----")
    val dfStep4 = dfStep3.na.fill(0.0, Seq("remote_allowed", "views", "applies", "zip_code", "fips", "company_id"))
    log("  Filled: remote_allowed, views, applies, zip_code, fips, company_id → 0.0")

    // Step 5: Remove salary outliers (keep 10k–1M or null)
    log("\n----- STEP 5: REMOVE SALARY OUTLIERS -----")
    log("  Keeping: normalized_salary between 10,000 and 1,000,000 (or null)")
    val dfStep5 = dfStep4.filter(
      col("normalized_salary").isNull ||
      (col("normalized_salary") >= 10000 && col("normalized_salary") <= 1000000)
    )
    log(s"  Rows after: ${dfStep5.count()}")

    // Step 6: Fill missing salary values (normalized_salary -> 0, min/max -> median)
    log("\n----- STEP 6: FILL MISSING SALARY VALUES (normalized -> 0, min/max -> median) -----")

    // (اختياري) ما نحتاج medNormalized لأننا بنعبي normalized_salary = 0
    val medMin = dfStep5.filter(col("min_salary").isNotNull)
      .stat.approxQuantile("min_salary", Array(0.5), 0.001)(0)

    val medMax = dfStep5.filter(col("max_salary").isNotNull)
      .stat.approxQuantile("max_salary", Array(0.5), 0.001)(0)

    val dfStep6a = dfStep5
      .withColumn(
        "normalized_salary",
        when(col("normalized_salary").isNull, lit(0.0))
          .otherwise(col("normalized_salary"))
      )
      .withColumn(
        "min_salary",
        when(col("min_salary").isNull, lit(medMin))
          .otherwise(col("min_salary"))
      )
      .withColumn(
        "max_salary",
        when(col("max_salary").isNull, lit(medMax))
          .otherwise(col("max_salary"))
      )

    log("  normalized_salary nulls -> 0.0")
    log(f"  min_salary median        = $medMin%.2f")
    log(f"  max_salary median        = $medMax%.2f")

    // Step 7: Fix salary errors (min > max)
    log("\n----- STEP 7: FIX SALARY ERRORS (min > max) -----")
    val dfStep6 = dfStep6a.filter(
      col("min_salary").isNull || col("max_salary").isNull ||
      col("min_salary") <= col("max_salary")
    )
    log(s"  Rows after: ${dfStep6.count()}")

    // Step 8: Deduplicate by job_id
    log("\n----- STEP 8: DEDUPLICATE BY job_id -----")
    val dfClean = dfStep6.dropDuplicates("job_id")
    dfClean.cache()
    val cleanCount = dfClean.count()
    log(s"  Rows after: $cleanCount")

    // ── Cleaning Summary ──────────────────────────────────────────
    log("\n----- CLEANING SUMMARY -----")
    log(f"  Rows before   : $rawCount%,d")
    log(f"  Rows after    : $cleanCount%,d")
    log(f"  Rows removed  : ${rawCount - cleanCount}%,d")
    log(f"  Columns before: ${dfRaw.columns.length}")
    log(f"  Columns after : ${dfClean.columns.length}")

    // ── Missing Values After Cleaning ─────────────────────────────
    log("\n----- MISSING VALUES AFTER CLEANING -----")
    var anyMissing = false
    dfClean.columns.foreach { c =>
      val missing = dfClean.filter(col(c).isNull || (col(c).cast("string") === "")).count()
      if (missing > 0) {
        anyMissing = true
        val pct = (missing.toDouble / cleanCount) * 100.0
        log(f"  $c%-35s missing=${missing}%8d (${pct}%.2f%%)")
      }
    }
    if (!anyMissing) log("  No missing values remaining ✓")

    // ── Sample Clean Data ─────────────────────────────────────────
    log("\n----- SAMPLE CLEAN ROWS (10) -----")
    dfClean.select("job_id", "title", "company_name", "location",
                   "formatted_work_type", "formatted_experience_level",
                   "normalized_salary", "remote_allowed")
      .limit(10).collect()
      .foreach { r =>
        def v(i: Int) = Option(r.get(i)).map(_.toString).getOrElse("NULL")
        log(s"  [${v(0)}] ${v(1).take(40)} | ${v(2).take(25)} | ${v(3).take(20)} | ${v(4)} | ${v(5)} | salary=${v(6)} | remote=${v(7)}")
      }

    // ── Salary Stats After Cleaning ───────────────────────────────
    log("\n----- SALARY STATS AFTER CLEANING -----")
    dfClean.filter(col("normalized_salary") > 0)
     .select(
        count("normalized_salary").alias("count"),
        round(avg("normalized_salary"), 2).alias("mean"),
        round(min("normalized_salary"), 2).alias("min"),
        round(max("normalized_salary"), 2).alias("max")
      ).collect().foreach { r =>
        log(s"  Count : ${r.get(0)}")
        log(s"  Mean  : ${r.get(1)}")
        log(s"  Min   : ${r.get(2)}")
        log(s"  Max   : ${r.get(3)}")
      }

    // ── Experience Level After Cleaning ───────────────────────────
    log("\n----- EXPERIENCE LEVEL AFTER CLEANING -----")
    dfClean.groupBy("formatted_experience_level").count()
      .orderBy(desc("count")).collect()
      .foreach(r => log(f"  ${r.getString(0)}%-25s ${r.getLong(1)}%,d"))

    // ── Remote Distribution After Cleaning ────────────────────────
    log("\n----- REMOTE DISTRIBUTION AFTER CLEANING -----")
    dfClean.groupBy("remote_allowed").count()
      .orderBy(desc("count")).collect()
      .foreach(r => log(f"  ${String.valueOf(r.get(0))}%-10s ${r.getLong(1)}%,d"))

    // ── SAVE CLEAN DATA TO CSV ────────────────────────
    sep("SAVE CLEAN DATA TO CSV")

    log(s"\n  Saving clean data to: $csvFinal")

    // Save temp
    dfClean
      .withColumn("job_id", col("job_id").cast("string"))
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("quoteAll", "true")
      .csv(csvTempDir)

    // Rename
    val tempDir = new File(csvTempDir)
    val partFile = tempDir.listFiles().filter(_.getName.startsWith("part-00000")).head
    Files.move(partFile.toPath, Paths.get(csvFinal), StandardCopyOption.REPLACE_EXISTING)

    // Remove temp
    tempDir.listFiles().foreach(_.delete())
    tempDir.delete()

    log(s"  Done! CSV saved as: $csvFinal")
    log(s"  Rows saved: $cleanCount")

    // ══════════════════════════════════════════════════════════════
    // PART 3 — DATA REDUCTION PIPELINE
    // ══════════════════════════════════════════════════════════════
    sep("PART 3: DATA REDUCTION PIPELINE")

    val reducedInputPath  = "src/main/resources/cleaned_linkedin_jobs.csv"
    val reducedTempDir    = "src/main/resources/reduced_linkedin_jobs_temp"
    val reducedFinalPath  = "src/main/resources/reduced_linkedin_jobs.csv"

    // ── (1) Load Cleaned Data ─────────────────────────────────────
    log("\n----- STEP 1: LOAD CLEANED DATA -----")
    val dfCleaned = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")
      .option("escape", "\"")
      .csv(reducedInputPath)
    val beforeReductionRows = dfCleaned.count()
    log(s"  Rows before reduction: $beforeReductionRows")
    log(s"  Columns before reduction: ${dfCleaned.columns.length}")

    // ── (2) FEATURE SELECTION: REMOVE IRRELEVANT METADATA ────────
    log("\n----- STEP 2: DROP IRRELEVANT METADATA COLUMNS -----")

    // FIX: removed job_id and company_name from drop list (they don't exist in cleaned CSV)
    val dfStepR1 = dfCleaned.drop(
      "job_posting_url",
      "application_url",
      "posting_domain",
      "expiry",
      "original_listed_time",
      "original_listed_time_ts",
      "listed_time_ts",
      "expiry_ts",
      "zip_code",
      "fips",
      "description",
      "pay_period",
      "views",
      "applies",
      "application_type",
      "listed_time",
      "sponsored",
      "work_type",
      "currency",
      "compensation_type",
      "company_id",
      "job_id",       
      "company_name" 
    )

    log("  Dropped: URLs, technical identifiers, and timestamp metadata")
    log(s"  Columns after Step 2: ${dfStepR1.columns.length}")

    // ── (3) REMOVE REDUNDANT SALARY COLUMNS ───────────────────────
    log("\n----- STEP 3: DROP REDUNDANT SALARY COLUMNS -----")

    val dfStepR2 = dfStepR1.drop("min_salary", "max_salary")

    log("  Dropped: min_salary, max_salary")
    log(s"  Columns after Step 3: ${dfStepR2.columns.length}")

    // ── (4) REDUCE LOCATION GRANULARITY (CITY → STATE) ────────────
    log("\n----- STEP 4: SIMPLIFY LOCATION TO STATE LEVEL -----")

    val dfStepR3 = dfStepR2.withColumn(
      "state",
      when(col("location").contains(","), trim(split(col("location"), ",")(1)))
        .otherwise(col("location"))
    )

    val dfReducedBase = dfStepR3.drop("location")

    log("  Extracted state from location column")
    log(s"  Columns after Step 4: ${dfReducedBase.columns.length}")

    // ── (5)  REMOVE ROWS WITHOUT VALID SALARY ────────────
    log("\n----- STEP 5: REMOVE ROWS WITHOUT VALID SALARY -----")

    val removeNullSalary = true

    val dfReduced =
      if (removeNullSalary) {
        val dfFiltered = dfReducedBase.filter(col("normalized_salary") > 0)
        log(s"  Rows after removing zero salaries: ${dfFiltered.count()}")
        dfFiltered
      } else {
        log("  Salary row removal skipped")
        dfReducedBase
      }
    val afterReductionRows = dfReduced.count()

    // ── Reduction Summary ─────────────────────────────────────────
    log("\n----- REDUCTION SUMMARY -----")
    log(f"  Rows before   : $beforeReductionRows%,d")
    log(f"  Rows after    : $afterReductionRows%,d")
    log(f"  Columns before: ${dfCleaned.columns.length}")
    log(f"  Columns after : ${dfReduced.columns.length}")

    // ── Sample Reduced Data ───────────────────────────────────────
    // FIX: updated select to only use columns that exist after reduction
    log("\n----- SAMPLE REDUCED ROWS (10) -----")
    dfReduced.select("title",
                     "state", "formatted_work_type",
                     "formatted_experience_level",
                     "normalized_salary", "remote_allowed")
      .limit(10)
      .collect()
      .foreach { r =>
        def v(i: Int) = Option(r.get(i)).map(_.toString).getOrElse("NULL")
        log(s"  ${v(0).take(35)} | ${v(1)} | ${v(2)} | ${v(3)} | salary=${v(4)} | remote=${v(5)}")
      }

    // ── SAVE REDUCED DATA TO CSV ──────────────────────────────────
    sep("SAVE REDUCED DATA TO CSV")

    log(s"\n  Saving reduced data to: $reducedFinalPath")

    dfReduced
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("quoteAll", "true")
      .csv(reducedTempDir)

    val reducedDir = new File(reducedTempDir)
    val reducedPartFile = reducedDir.listFiles().filter(_.getName.startsWith("part-00000")).head
    Files.move(reducedPartFile.toPath, Paths.get(reducedFinalPath), StandardCopyOption.REPLACE_EXISTING)

    reducedDir.listFiles().foreach(_.delete())
    reducedDir.delete()

    log(s"  Done! CSV saved as: $reducedFinalPath")
    log(s"  Rows saved: $afterReductionRows")

    // ══════════════════════════════════════════════════════════════
    // PART 4 — FEATURE TRANSFORMATION
    // ══════════════════════════════════════════════════════════════
    sep("PART 4: FEATURE TRANSFORMATION")

    // Load reduced data again to ensure clean start
    val dfPart4 = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(reducedFinalPath)

    log(s"  Rows loaded for transformation: ${dfPart4.count()}")
    log(s"  Columns: ${dfPart4.columns.mkString(", ")}")

    // ── Step 1: Transform categorical columns to numeric indices ──
    val categoricalCols = Seq(
      "title",
      "state",
      "formatted_work_type",
      "formatted_experience_level"
    )

    var dfTransformed = dfPart4

    categoricalCols.foreach { colName =>
      val idxCol = s"${colName}_idx_num"

      // Count frequency per title (for threshold) only for title column
      if (colName == "title") {
        val titleCounts = dfPart4.groupBy(colName).count()
        val threshold = 5 // Example threshold: rare titles <5 occurrences -> "Other"

        val rareTitles = titleCounts.filter(col("count") < threshold).select(col("title")).as[String].collect().toSet

        val udfReplaceRare = udf((t: String) => if (rareTitles.contains(t)) "Other" else t)
        dfTransformed = dfTransformed.withColumn("title_adj", udfReplaceRare(col("title")))

        val indexer = new StringIndexer()
          .setInputCol("title_adj")
          .setOutputCol(idxCol)
          .setHandleInvalid("keep")

        val model = indexer.fit(dfTransformed)
        dfTransformed = model.transform(dfTransformed).drop("title_adj")
      } else {
        val indexer = new StringIndexer()
          .setInputCol(colName)
          .setOutputCol(idxCol)
          .setHandleInvalid("keep")
        val model = indexer.fit(dfTransformed)
        dfTransformed = model.transform(dfTransformed)
      }

      log(s"  Column '$colName' transformed -> '$idxCol'")
    }

    // ── Step 2: Verify transformation ──
    log("\n----- SAMPLE TRANSFORMED ROWS (10) -----")
    dfTransformed.select(
      "title", "title_idx_num",
      "state", "state_idx_num",
      "formatted_work_type", "formatted_work_type_idx_num",
      "formatted_experience_level", "formatted_experience_level_idx_num",
      "normalized_salary", "remote_allowed"
    )
      .limit(10)
      .collect()
      .foreach { r =>
        def v(i: Int) = Option(r.get(i)).map(_.toString).getOrElse("NULL")
        log(s"  ${v(0).take(20)} | ${v(1)} | ${v(2)} | ${v(3)} | ${v(4)} | ${v(5)} | ${v(6)} | ${v(7)} | salary=${v(8)} | remote=${v(9)}")
      }

    // ── Step 3: Save transformed data for modeling ──
    val transformedPath = "src/main/resources/transformed_linkedin_jobs.csv"

    log(s"\n  Saving transformed data to: $transformedPath")

    dfTransformed
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("quoteAll", "true")
      .csv("src/main/resources/transformed_temp")

    // FIX: renamed to transformedTempDir / transformedPartFile to avoid conflict with Part 2
    val transformedTempDir = new File("src/main/resources/transformed_temp")
    val transformedPartFile = transformedTempDir.listFiles().filter(_.getName.startsWith("part-00000")).head
    Files.move(transformedPartFile.toPath, Paths.get(transformedPath), StandardCopyOption.REPLACE_EXISTING)
    transformedTempDir.listFiles().foreach(_.delete())
    transformedTempDir.delete()

    log(s"  Done! Transformed CSV saved as: $transformedPath")
    log(s"  Rows saved: ${dfTransformed.count()}")

// ── SAVE FIRST 20 ROWS SNAPSHOT ──────────────────────────────
sep("SAVE FIRST 20 ROWS SNAPSHOT")

val snapshotPath    = "src/main/resources/snapshot_first20_transformed.csv"
val snapshotTempDir = "src/main/resources/snapshot_temp"

log(s"\n  Saving first 20 rows snapshot to: $snapshotPath")

val dfSnapshotDF = dfTransformed.limit(20)

dfSnapshotDF
  .coalesce(1)
  .write
  .mode("overwrite")
  .option("header", "true")
  .option("quote", "\"")
  .option("escape", "\"")
  .option("quoteAll", "true")
  .csv(snapshotTempDir)

val snapshotDir = new File(snapshotTempDir)
val snapshotPartFile = snapshotDir.listFiles().filter(_.getName.startsWith("part-00000")).head
Files.move(snapshotPartFile.toPath, Paths.get(snapshotPath), StandardCopyOption.REPLACE_EXISTING)
snapshotDir.listFiles().foreach(_.delete())
snapshotDir.delete()

log(s"  Done! Snapshot saved as: $snapshotPath")
log(s"  Rows in snapshot: 20 (first rows of transformed data)")

    fw.close()
    println(s"\n Output saved to: $outputPath")
    spark.stop()
  }
}