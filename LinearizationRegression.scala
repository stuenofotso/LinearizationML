
/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */



/** Main class */
object LinearizationRegression {


  /** Main function */
  def main(args: Array[String]): Unit = {
    processLinearizationRegression(10000, 0.6)
  }

  case class Coord(x: Double) {
    def dist(c: Coord): Double = Math.abs(x - c.x)
  }
  class CoordOrdering(x: Coord) extends Ordering[Coord] {
    def compare(a: Coord, b: Coord): Int = a.dist(x) compare b.dist(x)
  }


  def top[T](xs: Seq[T], n: Int)(implicit ord: Ordering[T]): Seq[T] = {

    def insert[T](xs: Seq[T], e: T)(implicit ord: Ordering[T]): Seq[T] = {
      val (l, r) = xs.span(x => ord.lt(x, e))
      (l ++ (e +: r)).take(n)
    }

    xs.drop(n).foldLeft(xs.take(n).sorted)(insert)
  }


  //the real function
  def f(x:Double):Double = Math.sqrt(x)



  def processLinearizationRegression(N: Int, percentage:Double): Unit = {

    //parameters: must be fine tuned using training data
    val a = 2.0
    val b = 3.0
    val k = 10



    val data = scala.util.Random.shuffle(0 to N-1).splitAt(Math.round(N *percentage).toInt)

    val X = data._1.toArray
    val Y = X.map(f(_))
    val Yprime = X.map(t => a * t + b)

    val XTest = data._2.toArray
    val YTest = XTest.map(f(_))
    val YprimeTest = XTest.map(t => a * t + b)



    //overview of predictions
    scala.util.Random.shuffle(0 to X.size-1).take(20).map(i=>(i, predict(X(i), Y, Yprime, a, b, k))).foreach(t=>println("x = "+X(t._1)+", yre = "+Y(t._1)+" et ypre = "+t._2))


    //X.indices.foreach(i=>println(Y(i)+" 1:"+X(i)))
    //XTest.indices.foreach(i=>println(YTest(i)+" 1:"+XTest(i)))

    //X.indices.foreach(i=>println(X(i)+" "+Y(i)))
    //XTest.indices.foreach(i=>println(XTest(i)+" "+YTest(i)))




    println("Regression error = "+XTest.indices.par.aggregate(0.0)((s, i)=>s+Math.abs(predict(XTest(i), YTest, YprimeTest, a, b, k)-YTest(i)), _+_)/XTest.length)


  }

  def predict(x: Int, Y: Array[Double], Yprime:Array[Double], a:Double, b:Double, k:Int): Double = {

    val yprime = a * x + b


    //k-nearest neighbors
    val topYPrimeIndices = top(Yprime.map(Coord), k)(new CoordOrdering(Coord(yprime))).map(y=>Yprime.indexOf(y.x))

     topYPrimeIndices.aggregate(0.0)((s, j)=>s+((yprime*Y(j))/Yprime(j)), _+_)/topYPrimeIndices.size

  }


}