# Databricks notebook source
# MAGIC %md
# MAGIC #### LightGBM classifier
# MAGIC Run a lightGBM classifier, varying over max_depth and n_estimators (= numIterations)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Libraries needed
# MAGIC * Azure:mmlspark:0.15 (maven)
# MAGIC * mmlspark (python)

# COMMAND ----------

# MAGIC %md
# MAGIC Get max_depth/n_estimators from widgets

# COMMAND ----------

dbutils.widgets.text("n_estimators","100")
dbutils.widgets.text("max_depth","5")
dbutils.widgets.text("experiment_id","FILL_IN")
dbutils.widgets.text("run_id","FILL_IN")

n_estimators = int(dbutils.widgets.get("n_estimators"))
max_depth= int(dbutils.widgets.get("max_depth"))
experiment_id = dbutils.widgets.get("experiment_id")
run_id = dbutils.widgets.get("run_id")

# COMMAND ----------

# MAGIC %md
# MAGIC Load train and test data from global temporary views (defined in hyper Opte notebooks)

# COMMAND ----------

trainData = spark.sql("select * from global_temp.globalTempTrainData").repartition(sc.defaultParallelism) #  
testData = spark.sql("select * from global_temp.globalTempTestData").repartition(sc.defaultParallelism) #

# COMMAND ----------

# MAGIC %md
# MAGIC Train our LightGBM classifier model

# COMMAND ----------

from mmlspark import LightGBMClassifier
import pyspark.sql.functions as f

lgbClassifier = LightGBMClassifier(objective='binary',
                           maxDepth = max_depth,
                           numIterations=n_estimators,
                           labelCol = "label").fit(trainData)

# COMMAND ----------

import mlflow.lightgbm
import mlflow.spark

# COMMAND ----------

with mlflow.start_run(experiment_id = 3370548192880804) as run:
  mlflow.spark.log_model(lgbClassifier,"model")

# COMMAND ----------

mlflow_model_one = mlflow.spark.load_model("runs:/19933dd46bef4c4b85457ea4552fec08/model")

# COMMAND ----------

mlflow.spark.log_model(lgbClassifier,"modelTest")

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the light GBM classifier model on f1 score, and return the f1 score to the master notebook

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol = "prediction",labelCol="label", metricName="f1")

loss = 1 - evaluator.evaluate(lgbClassifier.transform(testData))

# COMMAND ----------

import mlflow
import mlflow.spark

mlflow.end_run() # end current active runs (can happen due to a potential error)
with mlflow.start_run(run_id = run_id, experiment_id = experiment_id) as run:
  mlflow.spark.log_model(lgbClassifier,"model")

# COMMAND ----------

dbutils.notebook.exit(str(loss))

# COMMAND ----------

model_1 = mlflow.spark.load_model("runs:/" + "e66fad95ca564a528bea8f2c9a03e4ac" + "/modelTest")

# COMMAND ----------

df_1 = model_1.transform(testData)

# COMMAND ----------

df_1.show()

# COMMAND ----------

