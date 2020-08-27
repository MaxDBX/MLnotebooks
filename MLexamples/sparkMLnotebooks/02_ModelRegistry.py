# Databricks notebook source
# MAGIC %md
# MAGIC ### Introduction
# MAGIC In our previous notebook we ran a number of models which are all tracked in the MLFlow experiment that we defined. In this notebook we want to log our best model to Model Registry (See **Models** on the left). If you have not done so yet, make sure to:
# MAGIC * go into your MLFlow experiment, select your best model based on test_f1 metric, and note down the run ID (you should see it at the top as RUN <RUN_ID> after clicking on the model)
# MAGIC * Copy the run ID into the runID text field at the top of the notebook.
# MAGIC * Lastly, also define a name for your model Registry by filling in the textfield for modelRegistryNAme

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Registry 
# MAGIC Only have to do this once (if you try to create again it will throw an error)

# COMMAND ----------

## Create ModelRegistry (only have to do this once)
client = MlflowClient()
client.create_registered_model(modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model associated with the selected runID to our model Registry

# COMMAND ----------

mlflow.register_model("runs:/" + runId + "/best_model", modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC You can now go to Models tab in the left, and click on the Registered Model with the name you previously defined. In it you will now see Version 1 of the model. However at this point it is not associated with any stage yet. In the below code we will push it to the production stage.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Push our model to production stage in our Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC Get versionID from the model registry that is associated with our runID. It could be that you see an `INVALID STATE` when running CMD 14. This simply means that the model has not been registered yet, and that you will need to wait for a few more seconds.

# COMMAND ----------

versionID = client.search_model_versions("run_id='%s'" % runId)[0].version

# COMMAND ----------

client.transition_model_version_stage(
    name=modelRegistryName,
    version=versionID,
    stage="production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC When you now go back to your registered model, you will see that your version 1 model is now classified as "Production". In the next notebook we will load the model from Production and put it in deployment!