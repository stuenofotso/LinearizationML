package timeusage

import java.lang.reflect.Parameter
import java.nio.file.Paths
import java.time.LocalDate

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}







/** Main class */
object TimeUsage {

  import org.apache.spark.sql.SparkSession
  import org.apache.spark.sql.functions._

  val spark: SparkSession =
    SparkSession
      .builder()
      .appName("Time Usage")
      .config("spark.master", "local")
      .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  /** Main function */
  def main(args: Array[String]): Unit = {
    //timeUsageByLifePeriod()

    val filePath = "/Users/steve/Downloads/sample_linear_regression_data.txt"

    println("multiLinearPerceptronClassifier")
    /*Test set accuracy = 0.6992518703241896 for data.txt*/
    /*Test set accuracy = 0.9019607843137255 for sample_multiclass_classification_data.txt*/
    /* Test set accuracy = 0.8888888888888888 for sample_binary_classification_data.txt*/
    /*Test set accuracy = 0.7972508591065293 for a1a.txt*/
    multiLinearPerceptronClassifier(filePath)

    println("logisticRegressionClassifier")
    logisticRegressionClassifier(filePath)


    println("randomForestClassifier")
    /* Test Error = 0.03053435114503822 for breast-cancer_scale.txt */
    randomForestClassifier(filePath)


  }

  def multiLinearPerceptronClassifier(filepath:String): Unit ={
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      //.load("/Users/steve/Downloads/data.txt")
      //.load("/Users/steve/Downloads/sample_multiclass_classification_data.txt")
      //.load("/Users/steve/Downloads/sample_binary_classification_data.txt")
      .load(filepath)

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    println("data overview "+train.head().mkString(", "))

    // specify layers for the neural network:
    // input layer of size _ (features), two intermediate of size 5 and 4
    // and output of size _ (classes)
    val layers = Array[Int](10, 5, 4, 2)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


  }



  def randomForestClassifier(filepath:String): Unit ={
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      //.load("/Users/steve/Downloads/data.txt")
      //.load("/Users/steve/Downloads/sample_multiclass_classification_data.txt")
      //.load("/Users/steve/Downloads/sample_binary_classification_data.txt")
      .load(filepath)

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (40% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.6, 0.4))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    // Test Error = 0.03053435114503822 for

  }




  def logisticRegressionClassifier(filePath:String): Unit ={
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      //.load("/Users/steve/Downloads/data.txt")
      //.load("/Users/steve/Downloads/sample_multiclass_classification_data.txt")
      //.load("/Users/steve/Downloads/sample_binary_classification_data.txt")
      .load(filePath)


    println("data overview "+data.head().mkString(", "))

    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(data)

    val trainingSummary = lrModel.summary

    val accuracy = trainingSummary.accuracy
    val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    val truePositiveRate = trainingSummary.weightedTruePositiveRate
    val fMeasure = trainingSummary.weightedFMeasure
    val precision = trainingSummary.weightedPrecision
    val recall = trainingSummary.weightedRecall
    println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
      s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")



  }



  def timeUsageByLifePeriod(): Unit = {
    val (columns, initDf) = read("/timeusage/atussum.csv")


    /* ADDED SPARK MLLIB CODE */


    println("ADDED SPARK MLLIB CODE")

    val mlDataFrame = initDf.drop(columns.head)
    mlDataFrame.printSchema()

    val mlDataRDD = mlDataFrame.rdd.map{ case row =>
      Vectors.dense(row.toSeq.toArray.map{
        x => x.asInstanceOf[Double]
      })
    }.cache()


    val numClusters = 10
    val numIterations = 20
    val clusters = KMeans.train(mlDataRDD, numClusters, numIterations)

    //clusters.predict()

    val WSSSE = clusters.computeCost(mlDataRDD)
    println("Within Set Sum of Squared Errors = " + WSSSE)


    return

    val (primaryNeedsColumns, workColumns, otherColumns) = classifiedColumns(columns)

    val summaryDf = timeUsageSummary(primaryNeedsColumns, workColumns, otherColumns, initDf)
    //summaryDf.show()
    //summaryDf.printSchema()

  //  val finalDf = timeUsageGrouped(summaryDf)
  //  finalDf.show()

  //  val finalDf1 = timeUsageGroupedSql(summaryDf)
  //  finalDf1.show()
//
    val finalDf2 = timeUsageGroupedTyped(timeUsageSummaryTyped(summaryDf))
    finalDf2.show()
  }

  /** @return The read DataFrame along with its column names. */
  def read(resource: String): (List[String], DataFrame) = {
    val rdd = spark.sparkContext.textFile(fsPath(resource))

    val headerColumns = rdd.first().split(",").to[List]
    // Compute the schema based on the first line of the CSV file
    val schema = dfSchema(headerColumns)

    val data =
      rdd
        .mapPartitionsWithIndex((i, it) => if (i == 0) it.drop(1) else it) // skip the header line
        .map(_.split(",").to[List])
        .map(row)


    val dataFrame =
      spark.createDataFrame(data, schema)





    (headerColumns, dataFrame)
  }

  /** @return The filesystem path of the given resource */
  def fsPath(resource: String): String =
    Paths.get(getClass.getResource(resource).toURI).toString

  /** @return The schema of the DataFrame, assuming that the first given column has type String and all the others
    *         have type Double. None of the fields are nullable.
    * @param columnNames Column names of the DataFrame
    */
  def dfSchema(columnNames: List[String]): StructType =
    if(columnNames.nonEmpty)
      StructType(StructField(columnNames.head.trim, StringType, nullable = false)::columnNames.tail.map(str=>StructField(str.trim, DoubleType, nullable = false)))
    else
      StructType(List[StructField]())



  /** @return An RDD Row compatible with the schema produced by `dfSchema`
    * @param line Raw fields
    */
  def row(line: List[String]): Row =
    if(line.nonEmpty){
      Row.fromSeq(line.head.trim ::line.tail.map(str=> str.trim.toDouble))
    }

    else
      Row(List[Any]())

  /** @return The initial data frame columns partitioned in three groups: primary needs (sleeping, eating, etc.),
    *         work and other (leisure activities)
    *
    * @see https://www.kaggle.com/bls/american-time-use-survey
    *
    * The dataset contains the daily time (in minutes) people spent in various activities. For instance, the column
    * “t010101” contains the time spent sleeping, the column “t110101” contains the time spent eating and drinking, etc.
    *
    * This method groups related columns together:
    * 1. “primary needs” activities (sleeping, eating, etc.). These are the columns starting with “t01”, “t03”, “t11”,
    *    “t1801” and “t1803”.
    * 2. working activities. These are the columns starting with “t05” and “t1805”.
    * 3. other activities (leisure). These are the columns starting with “t02”, “t04”, “t06”, “t07”, “t08”, “t09”,
    *    “t10”, “t12”, “t13”, “t14”, “t15”, “t16” and “t18” (those which are not part of the previous groups only).
    */
  def classifiedColumns(columnNames: List[String]): (List[Column], List[Column], List[Column]) = {
    val res = columnNames.groupBy(str=>
      if(str.matches("^t((0[13])|(1(1|(80[13])))).*")) 1
      else if(str.matches("^t((05)|(1805)).*")) 2
      else if(str.matches("^t((0[246789])|(1[0234568])).*")) 3
      else 4
    )
    (res(1).map(col), res(2).map(col), res(3).map(col))
  }

  /** @return a projection of the initial DataFrame such that all columns containing hours spent on primary needs
    *         are summed together in a single column (and same for work and leisure). The “teage” column is also
    *         projected to three values: "young", "active", "elder".
    *
    * @param primaryNeedsColumns List of columns containing time spent on “primary needs”
    * @param workColumns List of columns containing time spent working
    * @param otherColumns List of columns containing time spent doing other activities
    * @param df DataFrame whose schema matches the given column lists
    *
    * This methods builds an intermediate DataFrame that sums up all the columns of each group of activity into
    * a single column.
    *
    * The resulting DataFrame should have the following columns:
    * - working: value computed from the “telfs” column of the given DataFrame:
    *   - "working" if 1 <= telfs < 3
    *   - "not working" otherwise
    * - sex: value computed from the “tesex” column of the given DataFrame:
    *   - "male" if tesex = 1, "female" otherwise
    * - age: value computed from the “teage” column of the given DataFrame:
    *   - "young" if 15 <= teage <= 22,
    *   - "active" if 23 <= teage <= 55,
    *   - "elder" otherwise
    * - primaryNeeds: sum of all the `primaryNeedsColumns`, in hours
    * - work: sum of all the `workColumns`, in hours
    * - other: sum of all the `otherColumns`, in hours
    *
    * Finally, the resulting DataFrame should exclude people that are not employable (ie telfs = 5).
    *
    * Note that the initial DataFrame contains time in ''minutes''. You have to convert it into ''hours''.
    */
  def timeUsageSummary(
    primaryNeedsColumns: List[Column],
    workColumns: List[Column],
    otherColumns: List[Column],
    df: DataFrame
  ): DataFrame = {
    // Transform the data from the initial dataset into data that make
    // more sense for our use case
    // Hint: you can use the `when` and `otherwise` Spark functions
    // Hint: don’t forget to give your columns the expected name with the `as` method
    val workingStatusProjection: Column = when($"telfs" >= 1 && $"telfs" < 3, "working").otherwise("not working").as("working")
    val sexProjection: Column = when($"tesex" === 1, "male").otherwise("female").as("sex")
    val ageProjection: Column = when($"teage" >= 15 && $"teage" <= 22, "young").when($"teage" >= 23 && $"teage" <= 55, "active").otherwise("elder").as("age")

    // Create columns that sum columns of the initial dataset
    // Hint: you want to create a complex column expression that sums other columns
    //       by using the `+` operator between them
    // Hint: don’t forget to convert the value to hours

        val primaryNeedsProjection: Column = primaryNeedsColumns.reduce((c1, c2)=> c1 + c2)./(60).as("primaryNeeds")
        val workProjection: Column = workColumns.reduce((c1, c2)=> c1 + c2)./(60).as("work")
        //System.out.println("otherColumns "+otherColumns)
        val otherProjection: Column = otherColumns.reduce((c1, c2)=> c1 + c2)./(60).as("other")

    df
      .select(workingStatusProjection
        , sexProjection, ageProjection
    , primaryNeedsProjection
    , workProjection
    , otherProjection)
      .where($"telfs" <= 4) // Discard people who are not in labor force

  }

  /** @return the average daily time (in hours) spent in primary needs, working or leisure, grouped by the different
    *         ages of life (young, active or elder), sex and working status.
    * @param summed DataFrame returned by `timeUsageSumByClass`
    *
    * The resulting DataFrame should have the following columns:
    * - working: the “working” column of the `summed` DataFrame,
    * - sex: the “sex” column of the `summed` DataFrame,
    * - age: the “age” column of the `summed` DataFrame,
    * - primaryNeeds: the average value of the “primaryNeeds” columns of all the people that have the same working
    *   status, sex and age, rounded with a scale of 1 (using the `round` function),
    * - work: the average value of the “work” columns of all the people that have the same working status, sex
    *   and age, rounded with a scale of 1 (using the `round` function),
    * - other: the average value of the “other” columns all the people that have the same working status, sex and
    *   age, rounded with a scale of 1 (using the `round` function).
    *
    * Finally, the resulting DataFrame should be sorted by working status, sex and age.
    */
  def timeUsageGrouped(summed: DataFrame): DataFrame = {
    summed.cache().groupBy($"working",$"sex",$"age")
      .agg(round(avg($"primaryNeeds"), 1).as("primaryNeeds"),round(avg($"work"), 1).as("work"), round(avg($"other"), 1).as("other"))
      .orderBy("working", "sex", "age")
  }

  /**
    * @return Same as `timeUsageGrouped`, but using a plain SQL query instead
    * @param summed DataFrame returned by `timeUsageSumByClass`
    */
  def timeUsageGroupedSql(summed: DataFrame): DataFrame = {
    val viewName = s"summed"
    summed.cache().createOrReplaceTempView(viewName)
    spark.sql(timeUsageGroupedSqlQuery(viewName))
  }

  /** @return SQL query equivalent to the transformation implemented in `timeUsageGrouped`
    * @param viewName Name of the SQL view to use
    */
  def timeUsageGroupedSqlQuery(viewName: String): String =
  "SELECT working, sex, age, ROUND(AVG(primaryNeeds),1) AS primaryNeeds, ROUND(AVG(work),1) AS work, ROUND(AVG(other),1) AS other "+
  " FROM "+viewName+ " GROUP BY working, sex, age ORDER BY working, sex, age"

  /**
    * @return A `Dataset[TimeUsageRow]` from the “untyped” `DataFrame`
    * @param timeUsageSummaryDf `DataFrame` returned by the `timeUsageSummary` method
    *
    * Hint: you should use the `getAs` method of `Row` to look up columns and
    * cast them at the same time.
    */
  def timeUsageSummaryTyped(timeUsageSummaryDf: DataFrame): Dataset[TimeUsageRow] =
    timeUsageSummaryDf.cache().map(r=>TimeUsageRow(r.getAs[String]("working"),
      r.getAs[String]("sex"),
      r.getAs[String]("age"),
      r.getAs[Double]("primaryNeeds"),
      r.getAs[Double]("work"),
        r.getAs[Double]("other")))

  /**
    * @return Same as `timeUsageGrouped`, but using the typed API when possible
    * @param summed Dataset returned by the `timeUsageSummaryTyped` method
    *
    * Note that, though they have the same type (`Dataset[TimeUsageRow]`), the input
    * dataset contains one element per respondent, whereas the resulting dataset
    * contains one element per group (whose time spent on each activity kind has
    * been aggregated).
    *
    * Hint: you should use the `groupByKey` and `typed.avg` methods.
    */
  def timeUsageGroupedTyped(summed: Dataset[TimeUsageRow]): Dataset[TimeUsageRow] = {
    import org.apache.spark.sql.expressions.scalalang.typed
    summed.groupByKey(t=>(t.working, t.sex, t.age))
      .agg(
      round(typed.avg[TimeUsageRow](t=>t.primaryNeeds), 1).as("primaryNeeds").as[Double],
      round(typed.avg[TimeUsageRow](t=>t.work), 1).as("work").as[Double],
      round(typed.avg[TimeUsageRow](t=>t.other), 1).as("other").as[Double])
      .orderBy("key")
        .map(e=>TimeUsageRow(e._1._1, e._1._2, e._1._3, e._2, e._3, e._4))
  }
}

/**
  * Models a row of the summarized data set
  * @param working Working status (either "working" or "not working")
  * @param sex Sex (either "male" or "female")
  * @param age Age (either "young", "active" or "elder")
  * @param primaryNeeds Number of daily hours spent on primary needs
  * @param work Number of daily hours spent on work
  * @param other Number of daily hours spent on other activities
  */
case class TimeUsageRow(
  working: String,
  sex: String,
  age: String,
  primaryNeeds: Double,
  work: Double,
  other: Double
)