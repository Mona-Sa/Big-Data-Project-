import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.io.{FileWriter, BufferedWriter}

object SQLOperations {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("LinkedIn Jobs SQL Operations")
      .master("local[*]")
      .config("spark.driver.extraJavaOptions",
        "--add-opens=java.base/java.nio=ALL-UNNAMED " +
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
        "--add-opens=java.base/java.lang=ALL-UNNAMED " +
        "--add-opens=java.base/java.util=ALL-UNNAMED " +
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED " +
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED " +
        "--add-opens=java.base/java.io=ALL-UNNAMED " +
        "--add-opens=java.base/java.net=ALL-UNNAMED " +
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED " +
        "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED")
      .config("spark.executor.extraJavaOptions",
        "--add-opens=java.base/java.nio=ALL-UNNAMED " +
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
        "--add-opens=java.base/java.lang=ALL-UNNAMED " +
        "--add-opens=java.base/java.util=ALL-UNNAMED " +
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED " +
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED " +
        "--add-opens=java.base/java.io=ALL-UNNAMED " +
        "--add-opens=java.base/java.net=ALL-UNNAMED " +
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED " +
        "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val filePath   = if (args.length > 0) args(0) else "src/main/resources/transformed_linkedin_jobs.csv"
    val outputPath = if (args.length > 1) args(1) else "SQL_output.txt"

    val fw = new BufferedWriter(new FileWriter(outputPath))

    def log(line: String = ""): Unit = {
      println(line)
      fw.write(line + "\n")
    }

    def sep(): Unit = log("=" * 70)

    // ============================================================
    // LOAD DATASET & REGISTER TEMPORARY VIEW
    // Load the transformed dataset into a DataFrame and register
    // it as a temporary SQL view named "jobs" so that all queries
    // can reference it using standard Spark SQL syntax.
    // ============================================================
    log("Loading dataset and registering temporary view...")

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(filePath)

    df.createOrReplaceTempView("jobs")

    log(s"Dataset loaded: ${df.count()} rows, ${df.columns.length} columns")
    log(s"Temporary view registered as: jobs")
    log(s"Columns: ${df.columns.mkString(", ")}")
    sep()

    // ============================================================
    // QUERY 1
    // Question: What is the average normalized salary per experience
    // level, considering only experience levels with more than 100
    // job postings?
    // SQL Features: GROUP BY + HAVING
    // ============================================================
    log()
    sep()
    log(" QUERY 1: Average Salary per Experience Level (GROUP BY + HAVING)")
    sep()
    log("Question: What is the average normalized salary per experience")
    log("level, considering only experience levels with more than 100 job postings?")
    log()

    val query1 = spark.sql("""
      SELECT
        formatted_experience_level AS experience_level,
        ROUND(AVG(normalized_salary), 2)  AS avg_salary,
        COUNT(*)                          AS job_count
      FROM jobs
      GROUP BY formatted_experience_level
      HAVING COUNT(*) > 100
      ORDER BY avg_salary DESC
    """)

    log(f"  ${"Experience Level"}%-25s ${"Avg Salary"}%12s ${"Job Count"}%10s")
    log("-" * 55)
    query1.collect().foreach { row =>
      val expLevel  = Option(row.getString(0)).getOrElse("Unknown")
      val avgSalary = Option(row.get(1)).map(_.toString).getOrElse("N/A")
      val jobCount  = row.getLong(2)
      log(f"  $expLevel%-25s $avgSalary%12s $jobCount%10d")
    }

    sep()

    fw.close()
    println()
    println("Output saved to: " + outputPath)
    spark.stop()
  }
}