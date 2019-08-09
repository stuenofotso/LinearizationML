package multi_dimension

/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object MultiLinearizationRegression {


  /** Main function */
  def main(args: Array[String]): Unit = {
    processLinearizationRegression(10000, Util.M, 0.6)
  }


  def processLinearizationRegression(N: Int, M:Int, percentage: Double): Unit = {


    val data = Array.fill(N, M)(0.0)



    (0 until N).foreach(i=> data.update(i, scala.util.Random.shuffle(0 to M - 1).toArray.map(_.toDouble)))

    val dataTmp = data.splitAt(Math.round(N * percentage).toInt)

    val X = dataTmp._1.toArray
    val Y = X.map(Util.fr)
    val Yprime = X.map(Util.prime)

    val XTest = dataTmp._2.toArray
    val YTest = XTest.map(Util.fr)
    val YprimeTest = XTest.map(Util.prime)


    //overview of predictions
    scala.util.Random.shuffle(0 to X.length - 1).toList.take(20).map(i => (i, predict(X(i), Y, Yprime))).foreach(t => println("x = " + X(t._1).mkString(",") + ", yre = " + Y(t._1) + " et ypre = " + t._2))


    //X.indices.foreach(i=>println(Y(i)+ " "+ X(i).indices.map(j => (j + 1) + ":" + X(i)(j)).mkString(" ")))
    //XTest.indices.foreach(i=>println(YTest(i)+ " "+ XTest(i).indices.map(j => (j + 1) + ":" + XTest(i)(j)).mkString(" ")))




    println("XTest.length = "+XTest.length+" & Regression error = " + XTest.indices.par.aggregate(0.0)((s, i) => s + Math.abs(predict(XTest(i), Y, Yprime) - YTest(i)), _ + _) / XTest.length)


  }

  def predict(xs: Array[Double], Y: Array[Double], Yprime: Array[Double]): Double = {

    val yprime = Util.prime(xs)


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))

    topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * Y(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size

  }


}