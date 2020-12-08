# Databricks notebook source
# MAGIC %md
# MAGIC ### Get variables from the widgets

# COMMAND ----------

dbutils.widgets.text("runID","DefaultVal")
#dbutils.widgets.text("experimentID","1695779316543778")
#dbutils.widgets.text("modelRegistryName","mthoneBankXGB")

runId = dbutils.widgets.get("runID")
#experimentId = dbutils.widgets.get("experimentID")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import libraries

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Registry 
# MAGIC Only have to do this once

# COMMAND ----------

## Create ModelRegistry (only have to do this once)
client.create_registered_model(modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model associated to the runID to our model Registry

# COMMAND ----------

mlflow.register_model("runs:/" + runId + "/model", modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get versionID from model registry that is associated with our runID

# COMMAND ----------

versionID = client.search_model_versions("run_id='%s'" % runId)[0].version

# COMMAND ----------

# MAGIC %md
# MAGIC ### Push our model to production stage in our Model Registry

# COMMAND ----------

client.transition_model_version_stage(
    name=modelRegistryName,
    version=versionID,
    stage="production"
)