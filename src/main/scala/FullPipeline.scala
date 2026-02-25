import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.io.{FileWriter, BufferedWriter}

object FullPipeline {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("LinkedIn Jobs Full Pipeline")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val filePath = "src/main/resources/linkedin_jobs.csv"
    val outputPath = "full_pipeline_output.txt"
    
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
      .withColumn("listed_time_ts",      to_timestamp(from_unixtime(col("listed_time") / 1000)))
      .withColumn("original_listed_time_ts", to_timestamp(from_unixtime(col("original_listed_time") / 1000)))
      .withColumn("expiry_ts",           to_timestamp(from_unixtime(col("expiry") / 1000)))

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
    val dfStep3 = dfStep2.na.fill("Unknown", Seq("formatted_experience_level", "formatted_work_type"))
    log("  Filled: formatted_experience_level, formatted_work_type → 'Unknown'")

    // Step 4: Fill missing numeric values with 0
    log("\n----- STEP 4: FILL MISSING NUMERIC VALUES -----")
    val dfStep4 = dfStep3.na.fill(0.0, Seq("remote_allowed", "views", "applies"))
    log("  Filled: remote_allowed, views, applies → 0.0")

    // Step 5: Remove salary outliers (keep 10k–1M or null)
    log("\n----- STEP 5: REMOVE SALARY OUTLIERS -----")
    log("  Keeping: normalized_salary between 10,000 and 1,000,000 (or null)")
    val dfStep5 = dfStep4.filter(
      col("normalized_salary").isNull ||
      (col("normalized_salary") >= 10000 && col("normalized_salary") <= 1000000)
    )
    log(s"  Rows after: ${dfStep5.count()}")

    // Step 6: Remove rows where min_salary > max_salary
    log("\n----- STEP 6: FIX SALARY ERRORS (min > max) -----")
    val dfStep6 = dfStep5.filter(
      col("min_salary").isNull || col("max_salary").isNull ||
      col("min_salary") <= col("max_salary")
    )
    log(s"  Rows after: ${dfStep6.count()}")

    // Step 7: Deduplicate by job_id
    log("\n----- STEP 7: DEDUPLICATE BY job_id -----")
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
    dfClean.columns.foreach { c =>
      val missing = dfClean.filter(col(c).isNull || (col(c).cast("string") === "")).count()
      if (missing > 0) {
        val pct = (missing.toDouble / cleanCount) * 100.0
        log(f"  $c%-35s missing=${missing}%8d (${pct}%.2f%%)")
      }
    }

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
    dfClean.filter(col("normalized_salary").isNotNull)
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

    fw.close()
    println(s"\n Output saved to: $outputPath")
    spark.stop()
  }
}