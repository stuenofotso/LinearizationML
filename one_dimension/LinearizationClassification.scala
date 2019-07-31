package one_dimension


/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object LinearizationClassification {


  /** Main function */
  def main(args: Array[String]): Unit = {
    processLinearizationClassification(10000, 0.6)
  }


  def processLinearizationClassification(N: Int, percentage: Double) = {


    val data = scala.util.Random.shuffle(0 to N - 1).splitAt(Math.round(N * percentage).toInt)

    val X = data._1.toArray
    val Y = X.map(x => Util.fc(x.toDouble))
    val Yprime = X.map(x => Util.prime(x.toDouble))

    val XTest = data._2.toArray
    val YTest = XTest.map(x => Util.fc(x.toDouble))
    val YprimeTest = XTest.map(x => Util.prime(x.toDouble))


    //val x = scala.util.Random.nextInt(2 * N + 1)
    //println("x = "+x+", yre = "+f(x)+" et ypre = "+predict(x, Y, Yprime, a, b, k))


    //overview of predictions
    scala.util.Random.shuffle(0 to X.size - 1).take(20).map(i => (i, predict(X(i), Y, Yprime))).foreach(t => println("x = " + X(t._1) + ", yre = " + Y(t._1) + " et ypre = " + t._2))


    //X.indices.foreach(i=>println(Y(i)+" 1:"+X(i)))
    //XTest.indices.foreach(i=>println(YTest(i)+" 1:"+XTest(i)))


    //println("erreur de régression = "+XTest.indices.aggregate(0.0)((s, i)=>s+Math.pow(predict(XTest(i), YTest, YprimeTest, a, b, k)-YTest(i), 2), _+_))

    val predicts = XTest.indices.aggregate(0.0)((s, i) => if (predict(XTest(i), Y, Yprime) == YTest(i)) s + 1 else s, _ + _)

    println("Classification error = " + (XTest.length - predicts) / XTest.length + " well classified count = " + predicts + "/" + XTest.length)

  }

  def predict(x: Int, Y: Array[Double], Yprime: Array[Double]): Double = {

    val yprime = Util.prime(x)


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))

    if (topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * Y(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size >= 0.5) 1 else 0


  }


}