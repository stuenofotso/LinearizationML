package one_dimension


/*
@ author: Steve Tueno
@ email: stuenofotso@gmail.com
 */


/** Main class */
object LinearizationClassificationExp {
  //Classification using random probabilities


  //parameters: must be fine tuned using training data
  val pas = 0.01
  val error = 0.01
  val NumberConsensusPrior = 0.25
  val NiterMax = 10000

  /** Main function */
  def main(args: Array[String]): Unit = {
    processLinearizationClassificationExp(2000, 0.6)
  }


  /*
  @params:
  * YProba: valeur des probabilités estimée par la famille des k voisins
  * Y: labels
  *  YProba0:
   */
  def probabilityQuest(i: Int, X: Array[Double], Y: Array[Double], Yprime: Array[Double], YProba: Array[Double], allowedToViolate: List[Int], yProba0: Double, redo: Boolean): (Array[Double], Boolean) = {

    if (i >= X.length) {
      return (YProba, redo)
    }

    if (Math.abs(yProba0 - YProba(i)) >= error) {
      if ((Y(i) == 0.0 && yProba0 < 0.5) || (Y(i) == 1.0 && yProba0 <= 1.0 && yProba0 >= 0.5)) {
        YProba(i) = yProba0
      }
      else if (yProba0 - error > YProba(i) && ((Y(i) == 0.0 && YProba(i) + pas < 0.5) || (Y(i) == 1.0 && YProba(i) + pas <= 1.0))) {
        YProba(i) += pas
      }
      else if (yProba0 + error < YProba(i) && ((Y(i) == 0.0 && YProba(i) - pas >= 0.0) || (Y(i) == 1.0 && YProba(i) - pas >= 0.5))) {
        YProba(i) -= pas
      }
      else {
        if (allowedToViolate.contains(i)) {
          YProba(i) = yProba0
        }
        else {
          println("exception for a non allowed violation: Y(i) = " + Y(i) + " YProba(i)=" + YProba(i) + " yProba0=" + yProba0)
        }
      }


      //YProba(i) = yProba0


      if (i < X.length - 1) {
        return probabilityQuest(i + 1, X, Y, Yprime, YProba, allowedToViolate, learn(i + 1, X, Yprime, YProba), redo = true)
      }
      else {
        return (YProba, true)
      }
    }
    //    else if ((yProba0+pas) < YProba(i) && ((Y(i)==0.0&&yProba0>=0.0)||(Y(i)==1.0&&yProba0>=0.5))){
    //      YProba(i) = YProba(i)-pas
    //      probabilityQuest(i, X, Y, Yprime, YProba, learn(i, X, Y, Yprime, a, b, k), a, b, k, pas)
    //    }

    if (i < X.length - 1) {
      probabilityQuest(i + 1, X, Y, Yprime, YProba, allowedToViolate, learn(i + 1, X, Yprime, YProba), redo)
    }
    else {
      (YProba, redo)
    }


  }

  def learn(X: Array[Double], Y: Array[Double], Yprime: Array[Double], YProba: Array[Double], redo: Boolean, iter: Int): (Array[Double], Boolean) = {

    val oldProba = YProba.clone()


    val res = probabilityQuest(0, X, Y, Yprime, YProba, scala.util.Random.shuffle(0 to X.size - 1).take(Math.round(X.size * NumberConsensusPrior).toInt).toList, learn(0, X, Yprime, YProba), redo = false)

    println("res computed with redo status " + res._2)

    //(0 until X.length).foreach(i=>print(" Y(i)="+Y(i)+" proba(i)="+res._1(i))+"\t")
    //println()

    //if((0 until YProba.length).forall(i=>YProba(i)==res._1(i))){
    if (oldProba sameElements res._1) {
      println("stabilité détectée; interruption...")
      return res
    }

    //println(res._1.mkString(", "))

    if (res._2 && iter < NiterMax) {

      return learn(X, Y, Yprime, res._1, redo = false, iter + 1)
    }
    res
  }


  def processLinearizationClassificationExp(N: Int, percentage: Double) = {

    val data = scala.util.Random.shuffle(0 to N - 1).map(_.toDouble).splitAt(Math.round(N * percentage).toInt)

    val X = data._1.toArray
    val Y = X.map(Util.fc)
    val Yprime = X.map(Util.prime)
    val YProba0 = Y.map(y => if (y == 0.0) (scala.util.Random.nextDouble() * 2) / 5 else ((scala.util.Random.nextDouble() * 2) + 3) / 5)


    val XTest = data._2.toArray
    val YTest = XTest.map(Util.fc)
    //val YprimeTest = XTest.map(prime(_, a, b))


    //val x = scala.util.Random.nextInt(2 * N + 1)
    //println("x = "+x+", yre = "+f(x)+" et ypre = "+predict(x, Y, Yprime, a, b, k))


    //LEARNING PHASE
    val model = learn(X, Y, Yprime, YProba0, redo = false, 0)

    println("model computation ended. Redo = " + model._2)


    //PREDICTION/TESTING
    //overview of predictions
    scala.util.Random.shuffle(0 to X.size - 1).take(20).map(i => (i, predict(X(i), Yprime, model._1))).foreach(t => println("x = " + X(t._1) + ", yre = " + Y(t._1) + " et ypre = " + t._2))


    //X.indices.foreach(i=>println(Y(i)+" 1:"+X(i)))
    //XTest.indices.foreach(i=>println(YTest(i)+" 1:"+XTest(i)))


    val predictsX = X.indices.aggregate(0.0)((s, i) => if (predict(X(i), Yprime, model._1) == Y(i)) s + 1 else s, _ + _)

    println("Classification error on X = " + (X.length - predictsX) / X.length + " well classified count = " + predictsX + "/" + X.length)


    //println("erreur de régression = "+XTest.indices.aggregate(0.0)((s, i)=>s+Math.pow(predict(XTest(i), YTest, YprimeTest, a, b, k)-YTest(i), 2), _+_))

    val predicts = XTest.indices.aggregate(0.0)((s, i) => if (predict(XTest(i), Yprime, model._1) == YTest(i)) s + 1 else s, _ + _)

    println("Classification error on XTest = " + (XTest.length - predicts) / XTest.length + " well classified count = " + predicts + "/" + XTest.length)


  }

  def predict(x: Double, Yprime: Array[Double], YProba: Array[Double]): Double = {

    val yprime = Util.prime(x)


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))


    //topYPrimeIndices.map(Y).groupBy(identity).mapValues(_.size).maxBy(_._2)._1

    //if(topYPrimeIndices.map(Y).sum/topYPrimeIndices.size>0.5) 1.0 else 0.0

    val proba = topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * YProba(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size

    if (proba >= 0.5) 1.0
    else 0.0

  }


  def learn(i: Int, X: Array[Double], Yprime: Array[Double], YProba: Array[Double]): Double = {

    val yprime = Util.prime(X(i))


    //k-nearest neighbors
    val topYPrimeIndices = Util.top(Yprime.map(Util.Coord), Util.k)(new Util.CoordOrdering(Util.Coord(yprime))).map(y => Yprime.indexOf(y.x))

    //println("topYPrimeIndices of i="+i+" are "+topYPrimeIndices.mkString(","))


    //topYPrimeIndices.map(Y).groupBy(identity).mapValues(_.size).maxBy(_._2)._1

    //if(topYPrimeIndices.map(Y).sum/topYPrimeIndices.size>0.5) 1.0 else 0.0

    topYPrimeIndices.aggregate(0.0)((s, j) => s + ((yprime * YProba(j)) / Yprime(j)), _ + _) / topYPrimeIndices.size


  }


}