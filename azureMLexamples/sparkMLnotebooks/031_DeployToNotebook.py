# Databricks notebook source
# MAGIC %md
# MAGIC ### Deploy Spark ML model to Databricks notebook

# COMMAND ----------

# MAGIC %md
# MAGIC First off, we install the necessary libraries using dbutils.library utilities. The reason for this is that we want to run this notebook 
# MAGIC as an automated job that uses a new cluster. The easiest way to install libraries on a new job cluster is through using dbutils library utilities.

# COMMAND ----------

dbutils.library.installPyPI("mlflow") ## install library with notebook utils (this is convenient when running the job as an automated job)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Get model URI associated with our model in Model Registry production stage

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

dbutils.widgets.text("modelRegistryName","DefaultVal")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

def getProdModelURI(modelRegistryName):
  models = client.search_model_versions("name='%s'" % modelRegistryName)
  source = [model for model in models if model.current_stage == "Production"][0].source
  return source

modelURI = getProdModelURI(modelRegistryName)

# COMMAND ----------

modelURI

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Load our model and apply predictions
# MAGIC *Add in some guiding Text*

# COMMAND ----------

import mlflow.spark

# COMMAND ----------

## Why does this take 5 minutes???
spark_model = mlflow.spark.load_model(modelURI)

# COMMAND ----------

df = spark.sql("select * from max_db.bank_marketing_train_set")

# COMMAND ----------

resultDF = spark_model.transform(df.drop("label"))

# COMMAND ----------

display(resultDF.drop("features","rawPrediction"))

# COMMAND ----------

