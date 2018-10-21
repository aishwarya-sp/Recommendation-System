import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.hashing.MurmurHash3



object Aishwarya_SatishPadmanabhan_ModelBasedCF {

  var user_business_list = Map[Long,Set[Long]]()
  var user_business_rating_map = Map[(Long,Long),Double]()

  def get_user_average(uid: Long, bids: Set[Long]): Double = {
    var sum = 0.0
    val n = bids.size
    for(bid <- bids){
      val rating = user_business_rating_map(uid,bid)
      sum += rating

    }
    var avg = 0.0
    if (n== 0){
      avg = 0.0
    }
    else{
      avg = sum/n
    }
    avg

  }


  def range_of_value(value: Double): Int = {
    if (value >= 0.0 && value < 1.0) {
      0
    } else if(value >= 1.0 && value < 2.0) {
      1
    } else if(value >= 2.0 && value < 3.0) {
      2
    } else if(value >= 3.0 && value < 4.0) {
      3
    } else {
      4
    }
  }

  def hashFunc(str: String): Long = {
    val a = str.toCharArray.map(_.toByte)
    val hash1 = MurmurHash3.arrayHash(a)
    val hash2 = MurmurHash3.arrayHash(a.reverse)
    hash1 * hash2
  }

  def main(args: Array[String]): Unit = {

    // Load and parse the data

    val start = System.currentTimeMillis()

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("ModelBasedCF")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.OFF);
    val train_data = sc.textFile(args(0))
    val test_data = sc.textFile(args(1))


//    val train_data = sc.textFile("/Users/aishwaryasp/Desktop/train_review.csv")
//    val test_data = sc.textFile("/Users/aishwaryasp/Desktop/test_review.csv")

    //remove header row and split the line into fields user_id, business_id

    val input_data = train_data.filter(line => !line.contains("user_id"))
    val predict_data = test_data.filter(line => !line.contains("user_id"))

    val user_business_rating = input_data.map(_.split(',') match { case Array(user, item, rate) =>
      val  h3 = hashFunc(user)
      val h4 = hashFunc(item)
      ((h3,h4),rate.toDouble) })


    user_business_rating_map = user_business_rating.collect().toMap.map(identity)

    //create map for items rated by each user  user --> list of items reviews
    val user_business_ids = user_business_rating
      .keys.groupByKey()
      .mapValues(_.toSet)



    user_business_list = user_business_ids.collect().toMap



    val final_data = input_data.map(_.split(',') match { case Array(user, item, rate) =>

      val h1 = hashFunc(user).toInt
      val h2 = hashFunc(item).toInt
      Rating(h1,h2,rate.toDouble)
    })


    val user_test :RDD[(Long,String)] = predict_data.map(line => {

      val row = line.split(",")

      val h1 = hashFunc(row(0))

      (h1, row(0))

    })


    val user_map = user_test.collect().toMap.mapValues(_.toString).map(identity)


    val business_test :RDD[(Long,String)] = predict_data.map(line => {

      val row = line.split(",")

      val h1 = hashFunc(row(1))

      (h1, row(1))
    })

    val business_map = business_test.collect().toMap.mapValues(_.toString).map(identity)

    val test_set = predict_data.map(_.split(',') match { case Array(user, item, rate) =>
      val  h3 = hashFunc(user).toInt
      val h4 = hashFunc(item).toInt
      Rating(h3,h4,rate.toDouble)}
    )


    val rank = 2
    val numIterations = 25
    val alpha = 0.28
    val lambda = 0.3
//    val model = ALS.trainImplicit(final_data, rank, numIterations, lambda, alpha)
    val model = ALS.train(final_data, rank, numIterations, lambda)
    // Evaluate the model on rating data
    val usersProducts = test_set.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    val predict_set = predictions.map { case ((user, item), product) =>
      (user, item)
    }


    val new_users = usersProducts.subtract(predict_set)

    val new_user_ratings = new_users.map{ case (user,item) =>

      var rating = 3.5


      if(user_business_list.contains(user)){
        rating = get_user_average(user,user_business_list(user))
      }


      ((user,item),rating)
    }


   val final_set = predictions ++ new_user_ratings





    val ratesAndPreds = test_set.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(final_set)


    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
//    println(s"Mean Squared Error = $MSE")




    val difference = ratesAndPreds.map{ case ((user, product), (r1, r2)) =>
      (range_of_value(math.abs(r1 - r2)),1)}.countByKey()

    val tupledRDD = difference.map(line => (line._1, line._2)).toSeq.sortBy(_._1)

      for((k,v)<-tupledRDD) {
        if(k == 0) {
          println(">=0 and <1: "+ v)
        } else if (k == 1) {
          println(">=1 and <2: "+ v)
        } else if(k == 2) {
          println(">=2 and <3: "+ v)
        } else if(k == 3) {
          println(">=3 and <4: "+ v)
        } else {
          println(">=4: "+ v)
        }
      }


    val RMSE = math.sqrt(MSE)
    println(s"RMSE: $RMSE")


    val output_path = "Aishwarya_SatishPadmanabhan_ModelBasedCF.txt"
    val output = new PrintWriter(new File(output_path))

    val temp = final_set.map(elem=> ((user_map(elem._1._1), business_map(elem._1._2)), elem._2))

    val iter = temp.sortByKey().map(elem => elem._1._1 + "," + elem._1._2 + "," + elem._2).toLocalIterator
    while(iter.hasNext) {
      output.print(iter.next() + "\n")
    }
    output.close()
    val time = (System.currentTimeMillis()-start)/1000
    println("Time:  " + time + "sec")


  }

}
