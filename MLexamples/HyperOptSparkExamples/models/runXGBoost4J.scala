// Databricks notebook source
dbutils.widgets.text("max_depth","2")
dbutils.widgets.text("n_estimators","20")
dbutils.widgets.text("experiment_id","FILL_IN")
dbutils.widgets.text("run_id","FILL_IN")

// COMMAND ----------

val max_depth: Int = dbutils.widgets.get("max_depth").toInt
val n_estimators: Int = dbutils.widgets.get("n_estimators").toInt

// COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions",sc.defaultParallelism)

// COMMAND ----------

sc.defaultParallelism

// COMMAND ----------

// number of threads per xgb_worker. Should be set equal spark.tas.cpus. spark.task.cpus can be configured in cluster settings, but it's a cluster wide parameter.
// By default spark.task.cpus does not exist. Set it to 1 then.

// increasing n_threads can help a bit with memory management.
import java.util.NoSuchElementException
val n_threads = try {
  spark.conf.get("spark.task.cpus").toInt
} catch {
  case e: java.util.NoSuchElementException => 1
}

// number of "xgboost workers" to use. Pick this so that n_threads x xgb_workers = total # of cores (= sc.default parallelism)
val xgb_workers: Int = sc.defaultParallelism / n_threads   

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
val xgbClassifier = (new XGBoostClassifier().
                     setFeaturesCol("features").
                     setLabelCol("label").
                     setObjective("binary:logistic").
                     setMaxDepth(max_depth).
                     setNumRound(n_estimators).
                     setNumWorkers(xgb_workers).
                     setNthread(n_threads))

// COMMAND ----------

//  data partitions should be equal to xgb_workers
val trainData = spark.sql("select * from global_temp.globalTempTrainData").repartition(xgb_workers)
val testData = spark.sql("select * from global_temp.globalTempTestData").repartition(xgb_workers) 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Sparse vectors vs Dense Vectors
// MAGIC * When you make use of 0 values as missing values, you can pass the training/test data as sparse vectors to XGBoost4J.
// MAGIC * When you make use of a different value for missing, you need to convert your data to dense vectors using the command below.
// MAGIC * It is preferable to make use of sparse vectors for memory usage.

// COMMAND ----------

import org.apache.spark.sql.functions.{col, udf}
val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)

val denseTrainData = trainData.withColumn("features", toDense(col("features")))
val denseTestData = testData.withColumn("features",toDense(col("features")))

// COMMAND ----------

val xgbClassificationModel = xgbClassifier.fit(denseTrainData)

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = (new MulticlassClassificationEvaluator()
                 .setLabelCol("label")
                 .setPredictionCol("prediction")
                 .setMetricName("f1"))

// COMMAND ----------

val loss = 1 - evaluator.evaluate(xgbClassificationModel.transform(denseTestData))

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
val xgbClassificationModelPath = "/dbfs/tmp/carsten.thone@databricks.com/nativeModel"
xgbClassificationModel.nativeBooster.saveModel(xgbClassificationModelPath)

// COMMAND ----------

// MAGIC %python
// MAGIC import xgboost as xgb
// MAGIC import mlflow
// MAGIC import mlflow.xgboost
// MAGIC 
// MAGIC xgb_model = xgb.Booster({'nthread': 4})
// MAGIC xgb_model.load_model("/dbfs/tmp/carsten.thone@databricks.com/nativeModel")
// MAGIC experiment_id = dbutils.widgets.get("experiment_id")
// MAGIC run_id = dbutils.widgets.get("run_id")
// MAGIC 
// MAGIC with mlflow.start_run(run_id = run_id, experiment_id = experiment_id) as run:
// MAGIC   mlflow.xgboost.log_model(xgb_model,"model")

// COMMAND ----------

dbutils.notebook.exit(loss.toString)
