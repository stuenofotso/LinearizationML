package one_dimension


/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object LinearizationRegression {


  /** Main function */
  def main(args: Array[String]): Unit = {
    processLinearizationRegression(1000, 0.6)
  }


  def processLinearizationRegression(N: Int, percentage: Double): Unit = {


    val data = scala.util.Random.shuffle(0 to N - 1).splitAt(Math.round(N * percentage).toInt)

    val X = data._1.toArray
    val Y = X.map(x => Util.fr(x.toDouble))
    val Yprime = X.map(x => Util.prime(x.toDouble))

    val XTest = data._2.toArray
    val YTest = XTest.map(x => Util.fr(x.toDouble))
    val YprimeTest = XTest.map(x => Util.prime(x.toDouble))


    //overview of predictions
    scala.util.Random.shuffle(0 to X.size - 1).take(20).map(i => (i, predict(X(i), Y, Yprime))).foreach(t => println("x = " + X(t._1) + ", yre = " + Y(t._1) + " et ypre = " + t._2))


    //X.indices.foreach(i=>println(Y(i)+" 1:"+X(i)))
    //XTest.indices.foreach(i=>println(YTest(i)+" 1:"+XTest(i)))

    //X.indices.foreach(i=>println(X(i)+" "+Y(i)))
    //XTest.indices.foreach(i=>println(XTest(i)+" "+YTest(i)))


    println("Regression error = " + XTest.indices.par.aggregate(0.0)((s, i) => s + Math.abs(predict(XTest(i), Y, Yprime) - YTest(i)), _ + _) / XTest.length)


  }

  def predict(x: Int, Y: Array[Double], Yprime: Array[Double]): Double = {

    val yprime = Util.prime(x)


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))

    topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * Y(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size

  }


}