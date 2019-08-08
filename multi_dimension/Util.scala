package multi_dimension

/**
  * Created by steve on 30/07/2019.
  */
object Util {

  //parameters: must be fine tuned using training data
  val M=5
  val as: List[Double] = (1 to M).map(i=>Math.pow(i.toDouble,3)).toList
  val b = M+1.0
  val k = 5


  /*
  case class Coord(xs: Array[Double]) {
    def dist(c: Coord): Double = Math.sqrt(xs.zip(c.xs).aggregate(0.0)((s, x)=>s+Math.pow(x._1-x._2, 2), _+_))
  }

  def posPrime(n : Int)  =
for{
  i<- 2 until n
    j<-1 until i
    if isPrime(i+j)
} yield (i, j)


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
  */

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

  //the real classification function
  def fc(xs: Array[Double]): Double = if (fr(xs) % 2 > 1) 1 else 0

  //the real regression function
  def fr(xs: Array[Double]): Double = xs.aggregate(0.0, 0)((s, x)=>(s._1+Math.sqrt(x*s._2+1), s._2+1), (s1, s2)=>(s1._1+s2._1, s1._2+s2._2))._1


  def prime(xs: Array[Double]): Double = xs.indices.aggregate(0.0)((s, i)=>s+xs(i)*as(i), _+_) + b

}
