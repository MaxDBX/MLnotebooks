# Databricks notebook source
# MAGIC %md
# MAGIC ### Managing the model lifecycle with Model Registry
# MAGIC 
# MAGIC One of the primary challenges among data scientists in a large organization is the absence of a central repository to collaborate, share code, and manage deployment stage transitions for models, model versions, and their history. A centralized registry for models across an organization affords data teams the ability to:
# MAGIC 
# MAGIC * discover registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * transition models to deployment stages
# MAGIC * deploy different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * archive older models for posterity and provenance
# MAGIC * peruse model activities and annotations throughout model’s lifecycle
# MAGIC * control granular access and permission for model registrations, transitions or modifications
# MAGIC 
# MAGIC <div><img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry UI Workflows
# MAGIC The Model Registry UI is accessible from the Databricks workspace. From the Model Registry UI, you can conduct the following activities as part of your workflow:
# MAGIC 
# MAGIC * Register a model from the Run’s page
# MAGIC * Edit a model version description
# MAGIC * Transition a model version
# MAGIC * View model version activities and annotations
# MAGIC * Display and search registered models
# MAGIC * Delete a model version

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Registry API Workflows
# MAGIC All aspects of the Model Registry can be called programmatically via API. This notebook will show how an end-to-end workflow can be automated by only making use of this API. We will:
# MAGIC * Move any existing "production" model that's in our model registry to archived stage
# MAGIC * Move a new model associated with our runID (from an experiment we previously created) into the model Registry
# MAGIC * Move the new model to the "production" stage of the model registry

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Get variables from the widgets

# COMMAND ----------

dbutils.widgets.text("runID","DefaultVal")
dbutils.widgets.text("modelRegistryName","mthoneSingleNodeXGB")

# COMMAND ----------

runId = dbutils.widgets.get("runID")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Import libraries
# MAGIC The MlflowClient will be the main entry point for interacting with model registry API. We instantiate a `client` object with `MlflowClient()`, which contains a number of methods for interacting with the Model Registry, as we will see below.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Model Registry 
# MAGIC Only have to do this once. Note we can also do this through the UI in the Models tab.

# COMMAND ----------

## Create ModelRegistry (only have to do this once) TODO: Can probably delete this code
client = MlflowClient()
client.create_registered_model(modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Register the model associated to the runID to our model Registry
# MAGIC Note, we use `mlflow.register_model`. `register_model` is a function that's part of the `mlflow library`, and it's not a method that's part of the MLflow client.

# COMMAND ----------

"runs:/<run_id>/model"
mlflow.register_model("runs:/" + runId + "/model", modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Push models currently in production to archive stage (if it exists)
# MAGIC We use the `search_model_versions` method from our MLflow `client` object to get a list of models associated with our registered model. We then use a list comprehension to filter out the model versions for models that are in the Production stage. These versions are then moved to the archived stage.

# COMMAND ----------

model_list = client.search_model_versions("name='%s'" % modelRegistryName)

version_prod_list = [x.version for x in model_list if x.current_stage == "Production"]

for version in version_prod_list:
  client.transition_model_version_stage(
    name=modelRegistryName,
    version=version,
    stage="Archived")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Push our new model to the production stage in our model registry
# MAGIC Now that we have archived the old model, we can now push the new model to the production stage. First we get the model version id that is associated with the run ID. We then use this version ID to transition the model to production.

# COMMAND ----------

NewVersion = client.search_model_versions("run_id='%s'" % runId)[0].version

client.transition_model_version_stage(
    name=modelRegistryName,
    version=NewVersion,
    stage="production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Deploy the model to a databricks notebook
# MAGIC Now that the model is in the production stage, we can directly deploy it from there into a Databricks notebook, which is shown in the next notebook.
