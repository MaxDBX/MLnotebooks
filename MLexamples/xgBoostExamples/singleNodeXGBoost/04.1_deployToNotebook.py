# Databricks notebook source
dbutils.widgets.text("modelRegistryName","mthoneSingleNodeXGB")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

import mlflow

model_name = modelRegistryName
stage = "Production"

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{stage}")

# COMMAND ----------

# This is just the train data we used before but we "pretend" it's new data just to show how this works.
features = table('bank_db.bank_marketing_train_set').drop("label")
feature_cols = features.columns

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark,model_uri=f"models:/{model_name}/{stage}")
predictions = features.withColumn("prediction",loaded_model(*feature_cols))

# COMMAND ----------

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark,model_uri=f"models:/{model_name}/{stage}")
predictions = features.withColumn("prediction",loaded_model(*feature_cols))

# COMMAND ----------

display(predictions)