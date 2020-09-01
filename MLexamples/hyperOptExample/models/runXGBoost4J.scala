// Databricks notebook source
dbutils.widgets.text("max_depth","5")
dbutils.widgets.text("n_estimators","100")
dbutils.widgets.text("experiment_id","FILL_IN")
dbutils.widgets.text("run_id","FILL_IN")

// COMMAND ----------

val max_depth: Int = dbutils.widgets.get("max_depth").toInt
val n_estimators: Int = dbutils.widgets.get("n_estimators").toInt

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

val xgbClassifier = (new XGBoostClassifier().
                     setFeaturesCol("features").
                     setLabelCol("label").
                     setObjective("binary:logistic").
                     setMaxDepth(max_depth).
                     setNumRound(n_estimators).
                     setNumWorkers(sc.defaultParallelism).
                     setNthread(1).
                     setMissing(0.0f))

// COMMAND ----------

val trainData = spark.sql("select * from global_temp.globalTempTrainData").repartition(sc.defaultParallelism) //  
val testData = spark.sql("select * from global_temp.globalTempTestData").repartition(sc.defaultParallelism)   // set it equal to number of cores as a start

// COMMAND ----------

val xgbClassificationModel = xgbClassifier.fit(trainData)

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = (new MulticlassClassificationEvaluator()
                 .setLabelCol("label")
                 .setPredictionCol("prediction")
                 .setMetricName("f1"))

// COMMAND ----------

val loss = 1 - evaluator.evaluate(xgbClassificationModel.transform(testData))

// COMMAND ----------

// MAGIC %sh
// MAGIC ls /dbfs/tmp/carsten.thone@databricks.com/

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