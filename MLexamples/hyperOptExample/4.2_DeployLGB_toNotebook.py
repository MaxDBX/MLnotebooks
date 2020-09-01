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

# MAGIC %md
# MAGIC ### Step 2: Load our model and apply predictions
# MAGIC We can use the MLFlow spark flavor to load the LightGBM model. This may sound counterintuitive, however it works because mml spark LightGBM just uses Spark under the hood

# COMMAND ----------

import mlflow.lightgbm
import mlflow.spark
from mmlspark import LightGBMClassifier

# COMMAND ----------

LGB_model = mlflow.spark.load_model(modelURI)

# COMMAND ----------

df = spark.sql("select * from global_temp.globalTempTestData")
resultDF = LGB_model.transform(df)

# COMMAND ----------

display(resultDF)