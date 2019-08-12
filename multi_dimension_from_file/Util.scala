package multi_dimension_from_file

import org.apache.spark.sql.{Dataset, Encoder, Row, SparkSession}

/**
  * Created by steve on 30/07/2019.
  */
object Util {



  //parameters: must be fine tuned using training data
  val M=10
  //val as: List[Double] = (1 to M).map(i=>i.toDouble*2).toList
  //val b = M*10+3.0
  val as: List[Double] = (1 to M).map(i=>Math.pow(i.toDouble,3)).toList
  val b = M+1.0
  val k = 5


  val spark: SparkSession =
    SparkSession
      .builder()
      .appName("Time Usage")
      .config("spark.master", "local")
      .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")




  case class YPrime(count:Int, yre:Double, yprime:Double){
    def getYPrime(yp:Double): Double = if(count==1) ((yp * yre) / yprime) else yprime
  }

  case class YPrimeExp(count:Int, yre:Double, yprime:Double, yProba:Double, xStr:String, changed:Boolean) {

    def getYPrime(yp: Double): Double = if (count == 1) ((yp * yProba) / yprime) else yprime
    def dist(c: YPrimeExp): Double = Math.abs(yprime - c.yprime)

  }



  //implicit val yPrimeExpEncoder: Encoder[YPrimeExp] = org.apache.spark.sql.Encoders.kryo[YPrimeExp]


  class YPrimeExpOrdering(x: YPrimeExp) extends Ordering[YPrimeExp] {
    def compare(a: YPrimeExp, b: YPrimeExp): Int = a.dist(x) compare b.dist(x)
  }


  def top[T](xs: Seq[T], n: Int)(implicit ord: Ordering[T]): Seq[T] = {

    def insert[T](xs: Seq[T], e: T)(implicit ord: Ordering[T]): Seq[T] = {
      val (l, r) = xs.span(x => ord.lt(x, e))
      (l ++ (e +: r)).take(n)
    }

    xs.drop(n).foldLeft(xs.take(n).sorted)(insert)
  }


  def prime(x: Row): Double = x.getAs[org.apache.spark.ml.linalg.SparseVector](1).values.indices.aggregate(0.0)((s, i)=>s+x.getAs[org.apache.spark.ml.linalg.SparseVector](1).values(i)*as(x.getAs[org.apache.spark.ml.linalg.SparseVector](1).indices(i)), _+_)+b

}
