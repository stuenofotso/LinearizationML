package multi_dimension_from_file

import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row, SparkSession}

/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object MultiLinearizationClassification {


  /** Main function */
  def main(args: Array[String]): Unit = {
    //processLinearizationClassification("/Users/steve/Downloads/data.txt", 0.65)
    /*Classification error = 0.10265183917878529 well classified count = 3147/3507 for data.txt */
    //processLinearizationClassification("/Users/steve/Downloads/sample_multiclass_classification_data.txt", 0.6)
    /* Classification error = 0.35294117647058826 well classified count = 33/51 for sample_multiclass_classification_data.txt*/
    //processLinearizationClassification("/Users/steve/Downloads/sample_binary_classification_data.txt", 0.6)
    /*Classification error = 0.027777777777777776 well classified count = 35/36 for sample_binary_classification_data.txt M=692 */
    processLinearizationClassification("/Users/steve/Downloads/a1a.txt", 0.6)
    /* Classification error = 0.2697594501718213 well classified count = 425/582 for a1a.txt M=119 K=10 */


  }





  val spark: SparkSession =
    SparkSession
      .builder()
      .appName("Time Usage")
      .config("spark.master", "local")
      .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  import org.apache.spark.sql.functions._
  import spark.implicits._
  def processLinearizationClassification(filePath:String, percentage: Double) = {

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      .load(filePath)

    //data.take(50).foreach(head=>System.out.println("a row of data : "+head.mkString(", ")+" & "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices.mkString(", ")+" "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.mkString(", ")))

    // Split the data into train and test
    val splits = data.randomSplit(Array(percentage, 1-percentage), seed =  1234L)

    val X = splits(0).cache()
    //val X = data.sample(percentage)

    //X.take(50).foreach(head=>System.out.println("a row of X : "+head.mkString(", ")+" & "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices.mkString(", ")+" "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.mkString(", ")))


    //System.out.println("a row : "+X.head().mkString(", ")+" & "+X.head().length+" "+X.head().get(0).getClass+" "+X.head().get(1).getClass)
    //System.out.println("row values : "+X.head().getAs[org.apache.spark.ml.linalg.SparseVector](1).values.mkString(", "))



    val XTest = splits(1).cache()
    //val XTest = data.exceptAll(X)

    //XTest.take(50).foreach(head=>System.out.println("a row of XTest : "+head.mkString(", ")+" & "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices.mkString(", ")+" "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.mkString(", ")))

    val Yprime = X.map(row=>YPrime(1, row.getDouble(0), Util.prime(row))).cache()



    //X.indices.foreach(i=>println(Y(i)+ " "+ X(i).indices.map(j => (j + 1) + ":" + X(i)(j)).mkString(" ")))
    //XTest.indices.foreach(i=>println(YTest(i)+ " "+ XTest(i).indices.map(j => (j + 1) + ":" + XTest(i)(j)).mkString(" ")))


    //overview of predictions
    X.take(20).map(row => (row.mkString(","), predict(row, Yprime))).foreach(t => println("x = " + t._1  +  " et ypre = " + t._2))



    val wellClassifiedCount = XTest.collect().par.count(r => predict(r, Yprime) == r.getDouble(0))
    val countXtest = XTest.count()


    println("Classification error = " + (countXtest - wellClassifiedCount*1.0) / countXtest + " well classified count = " + wellClassifiedCount + "/" + countXtest)




  }

  def predict(x: Row,  Yprime: Dataset[YPrime]): Double = {

    val yprime = Util.prime(x)

    //println("Yprime: "+Yprime)
    //println("Yprime colums: "+ Yprime.take(20).mkString(", "))

    val topYPrimeIndices = Yprime.orderBy(udf((x:Double)=>Math.abs(yprime-x), DoubleType)($"yprime").asc).limit(Util.k)


    val res = topYPrimeIndices.reduce((r1, r2)=>YPrime(r1.count+r2.count, 0.0, r1.getYPrime(yprime) + r2.getYPrime(yprime)))

    if (res.yprime/ res.count >= 0.5) 1 else 0

    //println(" topYPrimeIndices "+ topYPrimeIndices)

    //k-nearest neighbors
    //val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord((x, yprime)))).map(_.x._1)

    //if (topYPrimeIndices.aggregate(0.0)((s, row) => s + ((yprime * row._2) / Yprime.filter($"_1".===(lit(row._1))).head()._2), _+_)  / topYPrimeIndices.size >= 0.5) 1 else 0


  }


}

case class YPrime(count:Int, yre:Double, yprime:Double){
  def getYPrime(yp:Double): Double = if(count==1) ((yp * yre) / yprime) else yprime
}