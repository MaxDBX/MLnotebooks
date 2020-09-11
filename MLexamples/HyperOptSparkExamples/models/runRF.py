# Databricks notebook source
dbutils.widgets.text("n_estimators","100")
dbutils.widgets.text("max_depth","5")
dbutils.widgets.text("experiment_id","FILL_IN")
dbutils.widgets.text("run_id","FILL_IN")

n_estimators = int(dbutils.widgets.get("n_estimators"))
max_depth= int(dbutils.widgets.get("max_depth"))
experiment_id = dbutils.widgets.get("experiment_id")
run_id = dbutils.widgets.get("run_id")

# COMMAND ----------

train_data = spark.sql("select * from global_temp.globalTempTrainData").repartition(sc.defaultParallelism)
test_data = spark.sql("select * from global_temp.globalTempTestData").repartition(sc.defaultParallelism)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

# COMMAND ----------

print(n_estimators)
print(max_depth)

rfHyperOpt = RandomForestClassifier(labelCol="label", 
                                    featuresCol="features", 
                                    maxDepth=max_depth, numTrees=n_estimators, 
                                    featureSubsetStrategy="all", seed=42, maxBins=100)

rfHyperOptFitted = rfHyperOpt.fit(train_data)

loss = 1 - evaluator.evaluate(rfHyperOptFitted.transform(test_data)) # 1 - f-score

# COMMAND ----------

import mlflow
import mlflow.spark
with mlflow.start_run(run_id = run_id, experiment_id = experiment_id) as run:
  mlflow.spark.log_model(rfHyperOptFitted,"model")

# COMMAND ----------

dbutils.notebook.exit(str(loss))

# COMMAND ----------

