package multi_dimension_from_file

import org.apache.spark.sql.{Dataset, Row}

/**
  * Created by steve on 30/07/2019.
  */
object Util {

  //parameters: must be fine tuned using training data
  val M=10
  val as: List[Double] = (1 to M).map(i=>i.toDouble*2).toList
  val b = M*10+3.0
  val k = 5




//  case class Coord(x: Row) {
//    def dist(c: Coord): Double = Math.abs(x.getDouble(1) - c.x.getDouble(1))
//  }
//
//  class CoordOrdering(x: Coord) extends Ordering[Coord] {
//    def compare(a: Coord, b: Coord): Int = a.dist(x) compare b.dist(x)
//  }
//
//
//  def top[T](xs: Seq[T], n: Int)(implicit ord: Ordering[T]): Seq[T] = {
//
//    def insert[T](xs: Seq[T], e: T)(implicit ord: Ordering[T]): Seq[T] = {
//      val (l, r) = xs.span(x => ord.lt(x, e))
//      (l ++ (e +: r)).take(n)
//    }
//
//    xs.drop(n).foldLeft(xs.take(n).sorted)(insert)
//  }
//

  def prime(x: Row): Double = x.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.indices.aggregate(0.0)((s, i)=>s+x.getAs[org.apache.spark.ml.linalg.SparseVector](1).values(i)*as(x.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices(i)), _+_)+b

}
