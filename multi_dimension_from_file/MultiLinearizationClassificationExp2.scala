package multi_dimension_from_file

import multi_dimension_from_file.Util.YPrimeExp
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
import multi_dimension_from_file.Util.spark.implicits._
import org.apache.spark.sql.functions._

object MultiLinearizationClassificationExp2 {
  //Classification using random probabilities


  //parameters: must be fine tuned using training data
  val pas = 0.01
  val error = 0.01
  val NumberConsensusPrior = 0.25
  val NiterMax = 10000

  //implicit val yPrimeExpEncoder: Encoder[YPrimeExp] = org.apache.spark.sql.Encoders.kryo[YPrimeExp]
  // alias for the type to convert to and from

  type YPrimeExpEncoded = (Int, Double, Double, Double, String, Boolean)

  // implicit conversions
  implicit def toEncoded(o: YPrimeExp): YPrimeExpEncoded = (o.count, o.yre, o.yprime, o.yProba, o.xStr, o.changed)
  implicit def fromEncoded(o: YPrimeExpEncoded): YPrimeExp =
     YPrimeExp(o._1, o._2, o._3, o._4, o._5, o._6)

  /** Main function */
  def main(args: Array[String]): Unit = {
    //processLinearizationClassificationExp("/Users/steve/Downloads/data.txt", 0.65)

    processLinearizationClassificationExp("/Users/steve/Downloads/cod-rna.txt", 0.6)
  }


  /*
  @params:
  * YProba: valeur des probabilités estimée par la famille des k voisins
  * Y: labels
  *  YProba0:
   */
  def probabilityQuest(x:Row, Yprime: List[YPrimeExp],  allowedToViolate: Array[String]): YPrimeExp = {

    //val Yprime = Yprime1.toDF("count", "yre", "yprime", "yProba", "xStr", "changed")



    val yprime = Yprime.par.filter(_.xStr==x.mkString(",")).head
    //println("yprime for "+yprime.xStr)
    val yProba0 = learn(x, Yprime)




    if (Math.abs(yProba0 - yprime.yProba) >= error) {
      if ((yprime.yre == 0.0 && yProba0 < 0.5) || (yprime.yre == 1.0 && yProba0 <= 1.0 && yProba0 >= 0.5)) {
        return YPrimeExp(1, yprime.yre, yprime.yprime, yProba0, yprime.xStr, changed = true)
      }
      else if (yProba0 - error > yprime.yProba && ((yprime.yre == 0.0 && yprime.yProba + pas < 0.5) || (yprime.yre == 1.0 && yprime.yProba + pas <= 1.0))) {
        return YPrimeExp(1, yprime.yre, yprime.yprime, yProba0+pas, yprime.xStr, changed = true)
      }
      else if (yProba0 + error < yprime.yProba && ((yprime.yre == 0.0 && yprime.yProba - pas >= 0.0) || (yprime.yre == 1.0 && yprime.yProba - pas >= 0.5))) {
        return YPrimeExp(1, yprime.yre, yprime.yprime, yProba0-pas, yprime.xStr, changed = true)
      }
      else {
        if (allowedToViolate.par.exists(_ == yprime.xStr)) {
          return YPrimeExp(1, yprime.yre, yprime.yprime, yProba0, yprime.xStr, changed = true)
        }
        else {
          println("exception for a non allowed violation: Y(i) = " + yprime.yre + " YProba(i)=" + yprime.yProba + " yProba0=" + yProba0)
          return YPrimeExp(1, yprime.yre, yprime.yprime, yprime.yProba, yprime.xStr, changed = false)
        }
      }
    }
    YPrimeExp(1, yprime.yre, yprime.yprime, yprime.yProba, yprime.xStr, changed = false)

  }



  @scala.annotation.tailrec
  def learn(X: DataFrame, Yprime: List[YPrimeExp], redo: Boolean, iter: Int): (Dataset[YPrimeExp], Boolean) = {

    //val oldProba = YProba.clone()

    val allowedToViolate = X.randomSplit(Array(NumberConsensusPrior, 1-NumberConsensusPrior))(0).map(_.mkString(",")).collect()

    //val res = Util.spark.sparkContext.parallelize(X.collect().par.map(x=>probabilityQuest(x, Yprime, allowedToViolate)).toList).toDS()

    //toDF(("count", "yre", "yprime", "yProba", "xStr", "changed")




    val res = X.map(x=>probabilityQuest(x, Yprime, allowedToViolate))




    if(res.filter($"changed".equalTo(lit(true))).count()>0) {
      if(iter<NiterMax){
        println("res computed with redo status true and NiterMax not reached... another try !")
        learn(X, res.collect().toList, true, iter+1)
      }
      else{
        println("res computed with redo status true but NiterMax is reached... gave up !")
        (res, true)
      }
    }
    else{
      println("stability detected; interruption...")
      (res, false)
    }
  }


  def processLinearizationClassificationExp(filePath:String, percentage: Double): Unit = {


    // Load the data stored in LIBSVM format as a DataFrame.
    val data = Util.spark.read.format("libsvm")
      .load(filePath)


    // Split the data into train and test
    val splits = data.randomSplit(Array(percentage, 1-percentage), seed =  1234L)

    val X = splits(0).cache()

    val XTest = splits(1).cache()
    //val XTest = data.exceptAll(X)

    //XTest.take(50).foreach(head=>System.out.println("a row of XTest : "+head.mkString(", ")+" & "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices.mkString(", ")+" "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.mkString(", ")))



    val Yprime = X.map(row=>YPrimeExp(1, row.getDouble(0), Util.prime(row), if (row.getDouble(0) == 0.0) (scala.util.Random.nextDouble() * 2) / 5 else ((scala.util.Random.nextDouble() * 2) + 3) / 5, row.mkString(","), changed = false)).collect().toList


    //LEARNING PHASE
    val model =  learn(X, Yprime, redo = false, 0)

    println("model computation ended. Redo = " + model._2)


    val modelList = model._1.collect().toList

    //PREDICTION/TESTING
    //overview of predictions
    X.take(20).map(row => (row.mkString(","), predict(row, modelList))).foreach(t => println("x = " + t._1  +  " et ypre = " + t._2))



    //println("schema of XTest.map(r => predict(r, modelList) == r.getDouble(0)) : "+XTest.map(r => predict(r, modelList) == r.getDouble(0)).printSchema())

    val wellClassifiedCount = XTest.map(r => predict(r, modelList) == r.getDouble(0)).filter($"value".equalTo(lit(true))).count()
    val countXtest = XTest.count()


    println("Classification error on XTest = " + (countXtest - wellClassifiedCount*1.0) / countXtest + " well classified count = " + wellClassifiedCount + "/" + countXtest)


    val wellClassifiedCountX = X.map(r => predict(r, modelList) == r.getDouble(0)).filter($"value".equalTo(lit(true))).count()
    val countX = X.count()


    println("Classification error on X = " + (countX - wellClassifiedCountX*1.0) / countX + " well classified count = " + wellClassifiedCountX + "/" + countX)

  }


  def predict(x: Row,  Yprime: List[YPrimeExp]): Double = {

    //println("compute prediction for "+x.mkString(","))

    val yprime = Util.prime(x)

    //println("Yprime: "+Yprime)
    //println("Yprime colums: "+ Yprime.take(20).mkString(", "))

    val topYPrimeIndices = Util.top(Yprime, Util.k)(new Util.YPrimeExpOrdering(Util.YPrimeExp(1, 0.0, yprime, 0.0, "", changed = false)))

    val res = topYPrimeIndices.par.reduce((r1, r2)=>YPrimeExp(r1.count+r2.count, 0.0, r1.getYPrime(yprime) + r2.getYPrime(yprime), 0.0, "", changed = false))

    //res.yprime/ res.count

    //val topYPrimeIndices = Yprime.orderBy(udf((x:Double)=>Math.abs(yprime-x), DoubleType)($"yprime").asc).limit(Util.k)

    //val res = topYPrimeIndices.reduce((r1, r2)=>YPrimeExp(r1.count+r2.count, 0.0, r1.getYPrime(yprime) + r2.getYPrime(yprime), 0.0, "", changed = false))

    if (res.yprime/ res.count >= 0.5) 1 else 0

  }



  def learn(x: Row,  Yprime: List[YPrimeExp]): Double = {

    val yprime = Util.prime(x)

    //println("Yprime: "+Yprime)
    //println("Yprime colums: "+ Yprime.take(20).mkString(", "))


    val topYPrimeIndices = Util.top(Yprime, Util.k)(new Util.YPrimeExpOrdering(Util.YPrimeExp(1, 0.0, yprime, 0.0, "", changed = false)))

    val res = topYPrimeIndices.par.reduce((r1, r2)=>YPrimeExp(r1.count+r2.count, 0.0, r1.getYPrime(yprime) + r2.getYPrime(yprime), 0.0, "", changed = false))

    res.yprime/ res.count


  }



}