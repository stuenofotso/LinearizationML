
/**
  * Created by steve on 30/07/2019.
  */
object Util {

  //parameters: must be fine tuned using training data
  val a = 2.0
  val b = 3.0
  val k = 3


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
  def fc(x: Double): Double = if (Math.sqrt(x) % 2 > 1) 1 else 0

  //the real regression function
  def fr(x: Double): Double = Math.sqrt(x)


  def prime(x: Double): Double = a * x + b

}
