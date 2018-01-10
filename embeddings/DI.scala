import scala.util.Try
import java.io._


case class cp( id: String, author: String, weight_tags: String )
case class cp_value( author: String, language_tag: String, scene_tag: String, style_tag: String, emotion_tag: String,theme_tag: String )
val cp_data = sc.textFile("/music/cp_data/idf_cp/20171223").filter(_.split("\t").length==26).map(x => {
			val list=x.split("\t")
			cp(list(0),list(6),list(25))//b_id,artist,weight_tags
		})

var cp_map = scala.collection.mutable.Map[String, cp_value]()

cp_data.collect.foreach(c => cp_map += ( c.id -> {
	val tagList=c.weight_tags.split("#")
	if(tagList.size==5)
		cp_value(c.author,tagList(0),tagList(1),tagList(2),tagList(3),tagList(4))
	else
		cp_value(c.author,"","","","","")
} ))

val broadcastCPmaps = spark.sparkContext.broadcast(cp_map)

val df_progress = spark.read.parquet("/user_profile/bi/data_xy_b_time_content/*/").
// val df_progress = time_content_df.
	select("insert_ts","p_uid","b_intent","b_source","b_id","b_progress","b_domain","b_source_type").
	filter("p_uid != '' and p_uid !=' '").
	groupBy("p_uid","b_id","b_intent","b_source","b_domain","b_source_type").
	agg(max("insert_ts").alias("insert_ts") ,max("b_progress").alias("b_progress")).
	select("p_uid","b_id","b_intent","b_source","insert_ts","b_progress","b_domain","b_source_type")

val up = df_progress.rdd.map(x => ((x(0).toString),(x(5).toString,x(1).toString,x(2).toString,x(3).toString,x(4).toString,x(6).toString,x(7).toString))).
	groupByKey().map(x => {
		(x._1,x._2.toList)
	})
def tryToInt( s: String ) = {import scala.util.Try;Try(s.toInt).toOption}

val up_middle = up.repartition(1).map(up => {
	var musics: List[String] = List()
	up._2.foreach(record=>{
		val progress = record._1
		val intent =record._3
		val b_id = record._2
		val b_source = record._4
		val b_domain = record._6
		val b_source_type = record._7
		if(broadcastCPmaps.value.contains(b_id)){//has cp
			val action_filter = (
        		intent=="next"
	          || intent=="stop"
	          || intent=="previous"
	          || intent=="play"
	          || intent=="play_collection"
	          || intent=="play_subscription"
	          || intent=="search_music"
	          || intent=="auto_next"
	          || intent=="search_music_without_slot"
	          || intent=="change_music"
	          || intent=="play_subscribe"
	        )
			if( action_filter && (
				b_source=="1" ||
				b_source=="2" ||
				b_source=="11")
				){
				val isPositive={
					if( tryToInt(progress).isDefined ){
						if (record._1.toInt > 10) { true }//positive
						else { false }//negtive
        			}else { false }
        		}

	        	//active musics
        		if(isPositive)
                    musics = musics :+ b_id
			}
		}
	})
	(musics)
}
).cache

val music_set = up_middle.flatMap(identity).distinct.zipWithIndex

var cp_index_map = scala.collection.mutable.Map[String, Long]()

music_set.collect.foreach(c => cp_index_map += ( c._1 -> c._2 ))

val music_index = up_middle.map(list => list.map( x => cp_index_map(x)) )

music_index.map(_.mkString("\t")).repartition(1).saveAsTextFile("/tmp/music_index_list")

music_set.map(x=> x._1+"\t"+x._2).repartition(1).saveAsTextFile("/tmp/music_set")


//val CONTENT_SIZE=4
//val word2vec_data = music_index.map( list:List[Long] => {
//    var x = ""
//    for(i <- CONTENT_SIZE/2 to list.length-CONTENT_SIZE/2 ){
//        for(j <- 0 to CONTENT_SIZE/2){
//            x = x+","+list[i+j]
//            x = x+","+list[i-j]
//        }
//        x+" "+list[i]
//    }
//})


