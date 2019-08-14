package multi_dimension_from_file

import multi_dimension_from_file.Util.YPrime
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.apache.commons.math3.stat.descriptive._


/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object MultiLinearizationRegressionMedian {


  /** Main function */
  def main(args: Array[String]): Unit = {
    //processLinearizationRegression("/Users/steve/Downloads/data_regression.txt", 0.6)
    /* XTest.length = 4010 & Regression error = 0.03009260609571779 data_regression.txt */
    processLinearizationRegression("/Users/steve/Downloads/data_tan.txt", 0.7)
  }


  import Util.spark.implicits._
  import org.apache.spark.sql.functions._
  def processLinearizationRegression(filePath:String, percentage: Double): Unit = {



    // Load the data stored in LIBSVM format as a DataFrame.
    val data = Util.spark.read.format("libsvm")
      .load(filePath)


    // Split the data into train and test
    val splits = data.randomSplit(Array(percentage, 1-percentage), seed =  1234L)

    val X = splits(0).cache()



    val XTest = splits(1).cache()

    val Yprime = X.map(row=>YPrime(1, row.getDouble(0), Util.prime(row))).cache()



    //overview of predictions
    X.take(20).map(row => (row.mkString(","), predict(row, Yprime))).foreach(t => println("x = " + t._1  +  " et ypre = " + t._2))



    val regressionError = XTest.collect().par.aggregate(0.0)((s, row) => s + Math.abs(predict(row, Yprime) - row.getDouble(0)), _ + _)

    val countXtest = XTest.count()


    println("XTest.length = "+countXtest+" & Regression error = " + (regressionError / countXtest))


  }

  def predict(x: Row,  Yprime: Dataset[YPrime]): Double = {

    val yprime = Util.prime(x)


    val topYPrimeIndices = Yprime.orderBy(udf((x:Double)=>Math.abs(yprime-x), DoubleType)($"yprime").asc).limit(Util.k)


    //val res = topYPrimeIndices.reduce((r1, r2)=>YPrime(r1.count+r2.count, 0.0, r1.getYPrime(yprime) + r2.getYPrime(yprime)))
    //estimer comme valeur médianne et pas comme moyenne

    //res.yprime/ res.count


    val arrMean = new DescriptiveStatistics()
    genericArrayOps(topYPrimeIndices.map(_.getYPrime(yprime)).collect()).foreach(v => arrMean.addValue(v))

    arrMean.getPercentile(50)


  }


}