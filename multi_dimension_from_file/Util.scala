package multi_dimension_from_file

import org.apache.spark.sql.{Dataset, Row}

/**
  * Created by steve on 30/07/2019.
  */
object Util {

  //parameters: must be fine tuned using training data
  val M=4
  val as: List[Double] = (1 to M).map(i=>Math.pow(i.toDouble,3)).toList
  val b = M+1.0
  val k = 4




  case class Coord(x: (Row, Double)) {
    def dist(c: Coord): Double = Math.abs(x._2 - c.x._2)
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



  def prime(X: Dataset[Row], yprime : Map[Row, Double]): Map[Row, Double] = {
    if(X.isEmpty) return yprime

    val head = X.head()


    //System.out.println("a row : "+head.mkString(", ")+" & "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices.mkString(", ")+" "+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.mkString(", "))

    prime(X.filter(_!=head), yprime+(head->(head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.indices.aggregate(0.0)((s, i)=>s+head.getAs[org.apache.spark.ml.linalg.SparseVector](1).values(i)*as(head.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices(i)), _+_)+b)))

  }

  def prime(x: Row): Double = x.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.indices.aggregate(0.0)((s, i)=>s+x.getAs[org.apache.spark.ml.linalg.SparseVector](1).values(i)*as(x.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices(i)), _+_)+b

}
