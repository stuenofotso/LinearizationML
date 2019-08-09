package multi_dimension

import org.apache.spark.sql.SparkSession

/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object MultiLinearizationClassification {


  /** Main function */
  def main(args: Array[String]): Unit = {
    processLinearizationClassification(10000, Util.M, 0.6)
  }


  def processLinearizationClassification(N: Int, M:Int, percentage: Double) = {

    val data = Array.fill(N, M)(0.0)



    (0 until N).foreach(i=> data.update(i, scala.util.Random.shuffle(0 to M - 1).toArray.map(_.toDouble)))

    val dataTmp = data.splitAt(Math.round(N * percentage).toInt)

    val X = dataTmp._1.toArray
    val Y = X.map(Util.fc)
    val Yprime = X.map(Util.prime)


    val XTest = dataTmp._2.toArray
    val YTest = XTest.map(Util.fc)

    //val x = scala.util.Random.nextInt(2 * N + 1)
    //println("x = "+x+", yre = "+f(x)+" et ypre = "+predict(x, Y, Yprime, a, b, k))


    //overview of predictions
    scala.util.Random.shuffle(0 to X.size - 1).toList.take(20).map(i => (i, predict(X(i), Y, Yprime))).foreach(t => println("x = " + X(t._1).mkString(",")  + ", yre = " + Y(t._1) + " et ypre = " + t._2))


    //X.indices.foreach(i=>println(Y(i)+ " "+ X(i).indices.map(j => (j + 1) + ":" + X(i)(j)).mkString(" ")))
    //XTest.indices.foreach(i=>println(YTest(i)+ " "+ XTest(i).indices.map(j => (j + 1) + ":" + XTest(i)(j)).mkString(" ")))




    val predicts = XTest.indices.aggregate(0.0)((s, i) => if (predict(XTest(i), Y, Yprime) == YTest(i)) s + 1 else s, _ + _)

    println("Classification error = " + (XTest.length - predicts) / XTest.length + " well classified count = " + predicts + "/" + XTest.length)

  }




  def predict(x: Array[Double], Y: Array[Double], Yprime: Array[Double]): Double = {

    val yprime = Util.prime(x)


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))

    if (topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * Y(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size >= 0.5) 1 else 0


  }


}