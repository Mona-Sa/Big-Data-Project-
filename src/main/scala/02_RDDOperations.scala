import org.apache.spark.sql.SparkSession
import java.io.{FileWriter, BufferedWriter, File}
import java.nio.file.{Files, Paths, StandardCopyOption}

object RDDOperations {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("LinkedIn Jobs RDD Analysis")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val sc = spark.sparkContext

    val filePath   = if (args.length > 0) args(0) else "src/main/resources/transformed_linkedin_jobs.csv"
    val outputPath = if (args.length > 1) args(1) else "RDD_output.txt"

    val fw = new BufferedWriter(new FileWriter(outputPath))

    def log(line: String = ""): Unit = {
      println(line)
      fw.write(line + "\n")
    }

    def sep(): Unit = log("=" * 70)

    def safeGet(row: Array[String], index: Int, default: String = "Unknown"): String = {
      if (index >= 0 && index < row.length) {
        val value = row(index).trim
        if (value.isEmpty) default else value
      } else default
    }

    def parseDouble(value: String): Double = {
      scala.util.Try(value.trim.toDouble).getOrElse(0.0)
    }

    def cleanExperienceLevel(value: String): String = {
      val cleaned = value.trim
      if (cleaned.isEmpty || cleaned == "0.0" || cleaned == "1.0") "Unknown" else cleaned
    }

    // ============================================================
    // LOAD DATASET
    // textFile loads the dataset and creates a raw string RDD.
    // We extract and skip the header, then split each row safely
    // handling commas inside quoted fields.
    // ============================================================
    val jobsRDD = sc.textFile(filePath)
    val header  = jobsRDD.first()

    val headerCols = header
      .split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
      .map(_.replace("\"", "").trim)

    val remoteIndex   = headerCols.indexOf("remote_allowed")
    val titleIndex    = headerCols.indexOf("title")
    val stateIndex    = headerCols.indexOf("state")
    val workTypeIndex = headerCols.indexOf("formatted_work_type")
    val expIndex      = headerCols.indexOf("formatted_experience_level")
    val salaryIndex   = headerCols.indexOf("normalized_salary")

    val dataRDD = jobsRDD.filter(row => row != header)

    val splitRDD = dataRDD.map(line =>
      line
        .split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
        .map(_.replace("\"", "").trim)
    )

    sep()
    log(" TRANSFORMATION: filter  |  ACTIONS: count + first")
    sep()

    // ─────────────────────────────────────────────────────
    // TRANSFORMATION: filter
    //
    // Filters the RDD to retain only remote jobs
    // (remote_allowed == "1.0").
    // This isolates a specific market segment that matters
    // when analyzing how work arrangement affects salary.
    // ─────────────────────────────────────────────────────
    log("TRANSFORMATION: filter")
    log("Extract rows where remote_allowed = 1 (Remote Jobs Only)")
    sep()

    val remoteJobsRDD = splitRDD.filter(row =>
      remoteIndex >= 0 && row.length > remoteIndex && safeGet(row, remoteIndex, "0.0") == "1.0"
    )

    // ─────────────────────────────────────────────────────
    // ACTION: count
    //
    // Counts how many remote jobs exist in the dataset.
    // ─────────────────────────────────────────────────────
    val remoteCount = remoteJobsRDD.count()
    val totalCount  = dataRDD.count()
    val percentage  = if (totalCount > 0) remoteCount.toDouble / totalCount * 100 else 0.0

    log(f"Total jobs in dataset : $totalCount%,d")
    log(f"Remote jobs (count)   : $remoteCount%,d")
    log(f"Remote percentage     : $percentage%.1f%%")
    log()

    // ─────────────────────────────────────────────────────
    // ACTION: first
    //
    // Retrieves the first record from the filtered remote RDD
    // to validate the filter transformation.
    // ─────────────────────────────────────────────────────
    if (remoteCount > 0) {
      val firstRemote = remoteJobsRDD.first()
      log("ACTION: first — Sample Remote Job Record")
      log("-" * 70)
      headerCols.zip(firstRemote).foreach { case (col, value) =>
        log(f"  $col%-35s : $value")
      }
    }

    sep()
    log("INTERPRETATION:")
    log("The filter transformation isolates remote job postings,")
    log(f"revealing that only ~$percentage%.1f%% of LinkedIn jobs offer remote work.")
    log("This helps us understand how common flexible work options are")
    log("before modeling salary patterns across roles and locations.")
    sep()

    log()
    sep()
    log(" TRANSFORMATION: map  |  ACTION: take(10)")
    sep()

    // ─────────────────────────────────────────────────────
    // TRANSFORMATION: map
    //
    // Transforms each row into:
    //   (state, salaryTier, workType, salary)
    //
    // This is feature engineering because it converts the raw
    // salary number into salary tiers that are easier to interpret.
    // ─────────────────────────────────────────────────────
    log("TRANSFORMATION: map")
    log("Each row -> (State, SalaryTier, WorkType, Salary)")
    sep()

    val mappedRDD = splitRDD.map(row => {
      val salary   = parseDouble(safeGet(row, salaryIndex, "0.0"))
      val state    = safeGet(row, stateIndex)
      val workType = safeGet(row, workTypeIndex)

      val salaryTier =
        if (salary < 60000) "Low"
        else if (salary < 100000) "Mid"
        else "High"

      (state, salaryTier, workType, salary)
    })

    // ─────────────────────────────────────────────────────
    // ACTION: take(10)
    //
    // Retrieves the first 10 mapped records.
    // ─────────────────────────────────────────────────────
    log("ACTION: take(10) — Preview first 10 mapped records")
    log(f"  ${"State"}%-35s ${"Tier"}%-6s ${"Work Type"}%-12s ${"Salary"}%s")
    log("-" * 70)

    mappedRDD.take(10).foreach { case (state, tier, workType, salary) =>
      log(f"  $state%-35s $tier%-6s $workType%-12s $$$salary%.0f")
    }

    sep()
    log("INTERPRETATION:")
    log("The map transformation derives a salaryTier feature from raw")
    log("salary values, helping us compare compensation patterns across")
    log("locations and work types in a cleaner and more interpretable way.")
    sep()

    log()
    sep()
    log(" TRANSFORMATION: groupByKey  |  ACTION: reduce")
    sep()

    // ─────────────────────────────────────────────────────
    // TRANSFORMATION: groupByKey
    //
    // Groups job titles by formatted experience level:
    //   (experienceLevel, title) -> groupByKey()
    //
    // This is useful for salary prediction because experience level
    // is strongly related to salary, and grouping titles under each
    // level shows how roles are distributed across seniority bands.
    // ─────────────────────────────────────────────────────
    log("TRANSFORMATION: groupByKey")
    log("Group job titles by formatted_experience_level")
    sep()

    val titlesByExperienceRDD = splitRDD
      .map(row => {
        val experience = cleanExperienceLevel(safeGet(row, expIndex))
        val title      = safeGet(row, titleIndex)
        (experience, title)
      })
      .groupByKey()

    log("Grouped job-title preview by experience level")
    log("-" * 70)

    titlesByExperienceRDD
      .mapValues(titles => titles.toSet.toList.sorted)
      .collect()
      .sortBy(_._1)
      .foreach { case (experience, titles) =>
        val preview = titles.take(3).map(_.take(25)).mkString(" | ")
        log(f"  $experience%-18s -> ${titles.size}%5d unique titles | Sample: $preview")
      }

    log()

    // ─────────────────────────────────────────────────────
    // ACTION: reduce
    //
    // Uses reduce to find the single record with the maximum
    // normalized_salary in the dataset.
    //
    // This directly supports our project goal because salary
    // prediction depends on identifying the upper salary bound
    // and the role/location combination associated with it.
    // ─────────────────────────────────────────────────────
    log("ACTION: reduce — Find record with maximum normalized_salary")
    log("-" * 70)

    val salaryRecordRDD = splitRDD.map(row => {
      val salary     = parseDouble(safeGet(row, salaryIndex, "0.0"))
      val title      = safeGet(row, titleIndex)
      val state      = safeGet(row, stateIndex)
      val experience = cleanExperienceLevel(safeGet(row, expIndex))
      val workType   = safeGet(row, workTypeIndex)
      (salary, title, state, experience, workType)
    })

    val maxSalaryRecord = salaryRecordRDD.reduce((a, b) => if (a._1 >= b._1) a else b)
    val maxSalary       = maxSalaryRecord._1

    log(f"  Maximum normalized salary : $$$maxSalary%.0f")
    log(s"  Title                     : ${maxSalaryRecord._2}")
    log(s"  State                     : ${maxSalaryRecord._3}")
    log(s"  Experience Level          : ${maxSalaryRecord._4}")
    log(s"  Work Type                 : ${maxSalaryRecord._5}")

    sep()
    log("INTERPRETATION:")
    log("The groupByKey transformation shows how different job titles")
    log("are distributed across experience levels, which helps explain")
    log("why salary varies by seniority and role. The reduce action")
    log("identifies the highest salary record in the dataset, giving us")
    log("a clear upper bound for salary prediction and a concrete example")
    log("of the role-location combination associated with top pay.")
    sep()

    log()
    sep()
    log(" TRANSFORMATION: reduceByKey  |  ACTION: saveAsTextFile")
    sep()

    // ─────────────────────────────────────────────────────
    // TRANSFORMATION: reduceByKey
    //
    // Computes the average normalized_salary per state by
    // summing salaries and counting records per state.
    // This aggregation reveals geographic salary patterns
    // which are directly relevant to salary prediction.
    // ─────────────────────────────────────────────────────
    log("TRANSFORMATION: reduceByKey")
    log("Compute average normalized_salary per state")
    sep()

    val stateSalaryRDD = splitRDD.map(row => {
      val state  = safeGet(row, stateIndex)
      val salary = parseDouble(safeGet(row, salaryIndex, "0.0"))
      (state, (salary, 1))
    })

    val stateTotalsRDD = stateSalaryRDD.reduceByKey((a, b) =>
      (a._1 + b._1, a._2 + b._2)
    )

    val stateAvgRDD = stateTotalsRDD.mapValues { case (total, count) =>
      total / count
    }

    log("Preview: Average salary per state (top 10)")
    log("-" * 70)
    stateAvgRDD.take(10).foreach { case (state, avg) =>
      log(f"  $state%-35s $$$avg%.0f")
    }

    // ─────────────────────────────────────────────────────
    // ACTION: saveAsTextFile
    //
    // Saves the (state, avgSalary) results to a CSV file
    // using the same coalesce + rename pattern used in the
    // preprocessing pipeline, producing a single clean file.
    // ─────────────────────────────────────────────────────
    val reduceTempDir  = "src/main/resources/reduceByKey_temp"
    val reduceFinalPath = "src/main/resources/reduceByKey_avg_salary_by_state.csv"

    stateAvgRDD
      .map { case (state, avg) => f"$state,$avg%.2f" }
      .coalesce(1)
      .saveAsTextFile(reduceTempDir)

    val reduceDir      = new File(reduceTempDir)
    val reducePartFile = reduceDir.listFiles().filter(_.getName.startsWith("part-00000")).head
    Files.move(reducePartFile.toPath, Paths.get(reduceFinalPath), StandardCopyOption.REPLACE_EXISTING)
    reduceDir.listFiles().foreach(_.delete())
    reduceDir.delete()

    log()
    log(s"ACTION: saveAsTextFile — Results saved to: $reduceFinalPath")
    sep()
    log("INTERPRETATION:")
    log("The reduceByKey transformation aggregates salary data by state,")
    log("revealing geographic compensation patterns across the US.")
    log("This supports salary prediction by confirming that location")
    log("is a meaningful factor in determining expected compensation.")
    sep()

    log()
    sep()
    log(" TRANSFORMATION: sortByKey  |  ACTION: saveAsTextFile")
    sep()

    // ─────────────────────────────────────────────────────
    // TRANSFORMATION: sortByKey
    //
    // Sorts the (state, avgSalary) pairs by average salary
    // in descending order to rank states by compensation.
    // This reveals which states offer the highest salaries,
    // providing insight into geographic salary distribution.
    // ─────────────────────────────────────────────────────
    log("TRANSFORMATION: sortByKey")
    log("Sort states by average salary (descending)")
    sep()

    val sortedRDD = stateAvgRDD
      .map { case (state, avg) => (avg, state) }
      .sortByKey(ascending = false)
      .map { case (avg, state) => (state, avg) }

    log("Top 10 states by average salary:")
    log("-" * 70)
    sortedRDD.take(10).foreach { case (state, avg) =>
      log(f"  $state%-35s $$$avg%.0f")
    }

    // ─────────────────────────────────────────────────────
    // ACTION: saveAsTextFile
    //
    // Saves the final sorted (state, avgSalary) results to
    // a CSV file using the same coalesce + rename pattern,
    // preserving the ranked compensation output as a single file.
    // ─────────────────────────────────────────────────────
    val sortedTempDir   = "src/main/resources/sortByKey_temp"
    val sortedFinalPath = "src/main/resources/sortByKey_top_states_by_salary.csv"

    sortedRDD
      .map { case (state, avg) => f"$state,$avg%.2f" }
      .coalesce(1)
      .saveAsTextFile(sortedTempDir)

    val sortedDir      = new File(sortedTempDir)
    val sortedPartFile = sortedDir.listFiles().filter(_.getName.startsWith("part-00000")).head
    Files.move(sortedPartFile.toPath, Paths.get(sortedFinalPath), StandardCopyOption.REPLACE_EXISTING)
    sortedDir.listFiles().foreach(_.delete())
    sortedDir.delete()

    log()
    log(s"ACTION: saveAsTextFile — Results saved to: $sortedFinalPath")
    sep()
    log("INTERPRETATION:")
    log("The sortByKey transformation ranks states by average salary,")
    log("identifying the highest-paying locations in the dataset.")
    log("This supports salary prediction by confirming that geographic")
    log("location is a strong determinant of compensation.")
    sep()

    fw.close()
    println()
    println("Output saved to: " + outputPath)
    spark.stop()
  }
}