import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}



object Aishwarya_SatishPadmanabhan_UserBasedCF {

  var user_business_rating_map = Map[(String,String),Double]()
  var user_business_list = Map[String,Set[String]]()
  var business_user_list= Map[String,Set[String]]()




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

  def get_user_average(uid: String, bids: Set[String]): Double = {
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

  def get_business_average(bid: String, uids: Set[String]): Double = {
    var sum = 0.0
    val n = uids.size
    for(uid <- uids){
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


  def calculate_pearson(l1: String, l2: String, b1:String): Double = {
    val bList1 = user_business_list(l1)
    val bList2 = user_business_list(l2)
    val commonList = bList1.intersect(bList2)
    val ruBar = get_user_average(l1,commonList)
    val rvBar = get_user_average(l2,commonList)

    var pearsonCoeff = 0.0

    var numeratorSum = 0.0
    var denoSum1 = 0.0
    var denoSum2 = 0.0

    if(commonList.isEmpty){
      pearsonCoeff = -1.0
    }

    else {

      for (i <- commonList) {
        numeratorSum += (user_business_rating_map(l1, i) - ruBar) * (user_business_rating_map(l2, i) - rvBar)
        denoSum1 += Math.pow((user_business_rating_map(l1, i) - ruBar), 2)
        denoSum2 += Math.pow((user_business_rating_map(l2, i) - rvBar), 2)
      }


      if (denoSum1 == 0 || denoSum2 == 0) {
        //avg of user
        pearsonCoeff = 0.0
        //        get_business_average(b1,business_user_list(b1))
      }
      else {
        pearsonCoeff = numeratorSum / (math.sqrt(denoSum1) * math.sqrt(denoSum2))
      }

    }

    pearsonCoeff

  }




  def predict_rating(uid:String, bid:String) : Double = {

    var prediction = 0.0
    var numSum = 0.0
    var denoSum = 0.0
    var count = 0
    var raBar = 0.0


    if(user_business_list.contains(uid) && business_user_list.contains(bid)){

      val userList = business_user_list(bid) diff Set(uid)
      raBar = get_user_average(uid, user_business_list(uid) -- Set(bid))
      for(user <- userList){
        val pc = calculate_pearson(uid,user,bid)
        val coratedItems = user_business_list(uid).intersect(user_business_list(user))
        val ruBar = get_user_average(user, coratedItems)

        denoSum += math.abs(pc)
        numSum += ((user_business_rating_map.getOrElse((user, bid), 2.5) - ruBar) * pc)

      }


      if(denoSum == 0.0 || (numSum/denoSum) < 0.0) {
        prediction = raBar
      } else {
        prediction = raBar + (numSum/denoSum)
      }

    }
    else if(user_business_list.contains(uid) && !business_user_list.contains(bid)){


       prediction = 3.5
//      prediction = get_user_average(uid,user_business_list(uid))

    }

    else if(!user_business_list.contains(uid) && business_user_list.contains(bid)){

      prediction = 3.5
//      prediction = get_business_average(bid,business_user_list(bid))

    }

    else{

      prediction = 2.5
    }


    if(prediction < 0.0) {
      prediction = 0.0
    } else if(prediction > 5.0) {
      prediction = 5.0
    }

    prediction

  }




  def main(args: Array[String]): Unit = {

    val start = System.currentTimeMillis()

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("ModelBasedCF")
    val sc = new SparkContext(conf)

    Logger.getLogger("org.apache.spark").setLevel(Level.OFF);
    val train = sc.textFile(args(0))
    val test = sc.textFile(args(1))

    //remove header row and split the line into fields user_id, business_id
    val header = train.first()
    val train_data = train.filter(line => !line.contains(header))
    val predict_data = test.filter(line => !line.contains(header))



    val user_business_rating = train_data.map(_.split(',') match { case Array(user, item, rate) =>

      ((user.toString,item.toString),rate.toDouble)
    })



    user_business_rating_map = user_business_rating.collect().toMap.map(identity)


    //create map for items rated by each user  user --> list of items reviews
    val user_business_ids = user_business_rating
      .keys.groupByKey()
      .mapValues(_.toSet)


    user_business_list = user_business_ids.collect().toMap

    //create map for user reviews for each item --> list of users who reviewed the item
    val business_user_ids = user_business_rating
      .keys.map(_.swap).groupByKey()
      .mapValues(_.toSet)

    business_user_list = business_user_ids.collect().toMap


    val test_set = predict_data.map(_.split(',') match { case Array(user, item, rate) =>

      val prate = predict_rating(user.toString, item.toString)

      ((user,item), (prate,rate.toDouble))
    })




    val MSE = test_set.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()



    val difference = test_set.map{ case ((user, product), (r1, r2)) =>
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
//    val time = (System.currentTimeMillis()-start)/1000
//    println("Time:  " + time + "sec")


    val output_path = "Aishwarya_SatishPadmanabhan_UserBasedCF.txt"
    val output = new PrintWriter(new File(output_path))

    val temp = test_set.map(elem=> ((elem._1._1, elem._1._2), elem._2._1))

    val iter = temp.sortByKey().map(elem => elem._1._1 + "," + elem._1._2 + "," + elem._2).toLocalIterator
    while(iter.hasNext) {
      output.print(iter.next() + "\n")
    }
    output.close()
//    println(s"Root Mean Squared Error= $RMSE")

    val time = (System.currentTimeMillis()-start)/1000
    println("Time:  " + time + "sec")
  }

}