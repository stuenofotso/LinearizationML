package one_dimension

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics

/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object LinearizationRegressionO2 {


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


    //overview of predictions
    //scala.util.Random.shuffle(0 to X.size - 1).take(20).map(i => (i, predict(X(i), Y, Yprime))).foreach(t => println("x = " + X(t._1) + ", yre = " + Y(t._1) + " et ypre = " + t._2))

    scala.util.Random.shuffle(0 to XTest.size - 1).take(20).map(i => (i, predict(XTest(i), Y, Yprime))).foreach(t => println("x = " + XTest(t._1) + ", yre = " + YTest(t._1) + " et ypre = " + t._2))


    //X.indices.foreach(i=>println(Y(i)+" 1:"+X(i)))
    //XTest.indices.foreach(i=>println(YTest(i)+" 1:"+XTest(i)))

    //X.indices.foreach(i=>println(X(i)+" "+Y(i)))
    //XTest.indices.foreach(i=>println(XTest(i)+" "+YTest(i)))


     //println("Regression error = " + XTest.indices.par.aggregate(0.0)((s, i) => s + Math.abs(predict(XTest(i), Y, Yprime) - YTest(i)), _ + _) / XTest.length)


  }

  def predict(x: Int, Y: Array[Double], Yprime: Array[Double]): Double = {

    val yprime = Util.prime(x)


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))



    //topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * Y(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size

    val yestims = topYPrimeIndices.map(j=> (j, (yprime * Y(j)) / Yprime(j))).sortBy(_._2).reverse

    val res1 = computePredictions(Y(yestims.head._1), Yprime(yestims.head._1), Y(yestims.tail.head._1), Yprime(yestims.tail.head._1), yestims.head._2, yestims.tail.head._2)
    val res2 = computePredictions(Y(yestims.tail.head._1), Yprime(yestims.tail.head._1), Y(yestims.tail.tail.head._1), Yprime(yestims.tail.tail.head._1), yestims.tail.head._2, yestims.tail.tail.head._2)


    println("results 1 "+res1.mkString(", "))
    println("results 2 "+res2.mkString(", "))

    //(res1 ++ res2).groupBy(x=>x).toList.maxBy(_._2.size)._1

    val arrMean = new DescriptiveStatistics()
    genericArrayOps((res1 ++ res2).toArray).foreach(v => arrMean.addValue(v))
    arrMean.getPercentile(50)
    //arrMean.getMean

  }

  def computePredictions(y1:Double, yprime1:Double, y2:Double, yprime2:Double, yestim1:Double, yestim2:Double):List[Double] = {
    val l1 = Math.sqrt(Util.d*Util.d+Math.pow(y1-yprime1, 2))
    val l2 = Math.sqrt(Util.d*Util.d+Math.pow(y2-yprime2, 2))
    val l1cosg1 = (l1*l1-l2*l2+Math.pow(yestim1-yestim2, 2))/(2*(yestim1-yestim2))
    val l2cosg2 = (l2*l2-l1*l1+Math.pow(yestim1-yestim2, 2))/(2*(yestim1-yestim2))

    val res1 = (l2*l2+2*(yestim2*l2cosg2 - yestim1*l1cosg1) - l1*l1+yestim2*yestim2 - yestim1*yestim1)/(-2*(yestim1+l1cosg1-yestim2-l2cosg2))
    val res2 = (l2*l2+2*(yestim2*l2cosg2 + yestim1*l1cosg1) - l1*l1+yestim2*yestim2 - yestim1*yestim1)/(-2*(yestim1-l1cosg1-yestim2-l2cosg2))
    val res3 = (l2*l2-2*(yestim2*l2cosg2 - yestim1*l1cosg1) - l1*l1+yestim2*yestim2 - yestim1*yestim1)/(-2*(yestim1-l1cosg1-yestim2+l2cosg2))

    List[Double](res1, res2, res3)
  }


}