# Databricks notebook source
import uuid
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
userhome = f"dbfs:/tmp/{username}"
useruuid = uuid.uuid4()

# COMMAND ----------

import uuid
uniqueId = uuid.uuid4()
spark.conf.set("com.databricks.tmp.uniqueid", str(uniqueId))

# COMMAND ----------

def setOrCeateMLFlowExperiment(experimentPath):
  from mlflow.exceptions import MlflowException
  try:
    experiment_id = mlflow.create_experiment(experimentPath)
  except MlflowException: # if experiment is already created
    experiment_id = mlflow.get_experiment_by_name(experimentPath).experiment_id
    mlflow.set_experiment(experimentPath)
  return experiment_id

# COMMAND ----------


