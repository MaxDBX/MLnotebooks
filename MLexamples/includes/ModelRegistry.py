# Databricks notebook source
# MAGIC %md
# MAGIC ### Get variables from the widgets

# COMMAND ----------

dbutils.widgets.text("runID","DefaultVal")
dbutils.widgets.text("modelRegistryName","mthoneSingleNodeXGB")

runId = dbutils.widgets.get("runID")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import libraries

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Registry 
# MAGIC Only have to do this once

# COMMAND ----------

client = MlflowClient()
# Register the model with the run URI and unique name
#model_uri = "runs:/{}/model".format(runId)
#model_registry_id = spark.conf.get("com.databricks.tmp.uniqueid") # ensures we have a uniqueID for each student
#model_registry_name = "bank_singlenode_{}".format(model_registry_id) 

#model_details = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

# COMMAND ----------

## Create ModelRegistry (only have to do this once) TODO: Can probably delete this code
client = MlflowClient()
client.create_registered_model(modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model associated to the runID to our model Registry

# COMMAND ----------

mlflow.register_model("runs:/" + runId + "/model", modelRegistryName)

# COMMAND ----------

# DBTITLE 1,Push old model(s) to archive stage if it exists
model_list = client.search_model_versions("name='%s'" % modelRegistryName)

version_prod_list = [x.version for x in model_list if x.current_stage == "Production"]

for version in version_prod_list:
  client.transition_model_version_stage(
    name=modelRegistryName,
    version=version,
    stage="Archived")

# COMMAND ----------

model_list = client.search_model_versions("name='%s'" % modelRegistryName)

# COMMAND ----------

# DBTITLE 1,Get versionID from model registry that is associated with our run ID.
NewVersion = client.search_model_versions("run_id='%s'" % runId)[0].version

# COMMAND ----------

# DBTITLE 1,Pushing our model to production stage in our model registry
client.transition_model_version_stage(
    name=modelRegistryName,
    version=NewVersion,
    stage="production"
)