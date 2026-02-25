ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "citibike-bigdata",
    version := "0.1.0-SNAPSHOT",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.5.1",
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
    Compile / mainClass := Some("FullPipeline"),
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq("--add-exports=java.base/sun.nio.ch=ALL-UNNAMED")
  )
