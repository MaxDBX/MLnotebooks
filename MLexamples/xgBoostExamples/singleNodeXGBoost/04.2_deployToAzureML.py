# Databricks notebook source
# MAGIC %md
# MAGIC ## Azure ML for Python
# MAGIC In this notebook we will use MLFLow and the azureml sdk for Python to:
# MAGIC * Load a model from MLFlow model registry.
# MAGIC * Add this model to AzureML model registry.
# MAGIC * (Attach a AKS cluster to AzureML)
# MAGIC * Deploy the model to an AKS cluster as a AzureML web service
# MAGIC * Run predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0. Prerequisites
# MAGIC Make sure you have the following Azure resources set up.
# MAGIC * AzureML workspace
# MAGIC * Azure Kubernetes Service (AKS) Inference Cluster within the AzureML workspace
# MAGIC * Azure Service Principal with Contributor access to the AzureML workspace 
# MAGIC 
# MAGIC ##### Libraries:
# MAGIC * Install azureml-mlflow on the databricks cluster. This will install the Azure ML sdk for python

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Retrieve AzureML workspace
# MAGIC To deploy a model to AzureML, we need to be able to interact with the AzureML workspace. To do this we need to retrieve a AzureML workspace object using the `Workspace` class. For authentication, we use a service principal object that we get through `ServicePrincipalAuthentication`, in which we fill in the credentials of a service principal (that you should have set up in Azure).

# COMMAND ----------

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

# COMMAND ----------

# svp credentials (BEST PRACTICE: MAKE USE OF DATABRICKS SECRETS BACKED BY SECRET SCOPE)

# General Azure variables
tenant_id = "cdd620fd-d435-4742-9eb0-956c1a84f986"
subscription_id = "718f7edc-ce1c-431b-9b68-7ac7d34415f4"
resource_group = "rg-azureml"


# SVP credentials
svp_id = "2cef2e6a-5306-447c-8c6a-149e1a2fe5c3"
pw = "cRR7Q~B4JGGAIZebHmmSJIpARkxuUytKZ5GKt"

# Azure ML workspace name
aml_workspace = "mthonedev"

# COMMAND ----------

# DBTITLE 1,Retrieve Service Principal Object
## Get service principal object
svc_pr = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=svp_id,
    service_principal_password=pw)

# COMMAND ----------

# DBTITLE 1,Retrieve AzureML object
ws = Workspace(
  subscription_id = subscription_id,
  resource_group = resource_group,
  workspace_name = aml_workspace,
  auth = svc_pr
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Provision or retrieve AKS cluster associated with Azure ML Workspace
# MAGIC We already created an AKS cluster using the AzureML UI. All we need to do is simply retrieve it. If you want to provision an AKS cluster to the ML workspace, uncomment the lines below.

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget

# COMMAND ----------

# prov_config = AksCompute.provisioning_configuration # default configuration
# aks_name = "mthone-aks"
# aks_target = ComputeTarget.create(workspace = ws,
#                                   name = aks_name,
#                                   provisioning_configuration = prov_config)

compute_target = ComputeTarget(workspace = ws, name = "mthone-aks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create an AzureML environment
# MAGIC From the [docs](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments): _Azure Machine Learning environments are an encapsulation of the environment where your machine learning training happens. They specify the Python packages, environment variables, and software settings around your training and scoring scripts. They also specify run times (Python, Spark, or Docker). _
# MAGIC 
# MAGIC Here we are going to specify a python runtime, using a `conda.yml` file.

# COMMAND ----------

from azureml.core import Environment

# COMMAND ----------

requirements_script = "dbfs:/tmp/mthone/env.yml"

dbutils.fs.put(requirements_script,""" 
name: test-env
dependencies:
- python=3.7.3
- pip
- pip:
  - xgboost
  - azureml-defaults
  - sklearn
  - pandas
  - numpy
  - pickle5
  - mlflow""", overwrite = True)

env = Environment.from_conda_specification("deploy_env", "/dbfs/tmp/mthone/env.yml")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Create an Entry Script
# MAGIC The entry script is a python script that lives in the Azure ML compute cluster, and basically loads the model and runs predictions. It consists of an `init()` method, in which we define variables that should be loaded on start-up, such as loading our ML model. in `run()` we define how we are going to handle incoming data. This is where we generate the predictions.

# COMMAND ----------

entry_script_name = "dbfs:/tmp/mthone/model_score.py"

dbutils.fs.put(entry_script_name, """
import json
import pandas as pd
import os
import pickle5 as pickle
import mlflow

def init():
  global model
  model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
  model = pickle.load(open(model_path,"rb"))
  print("model is loaded")
  
def run(data):
  try:
    #json_data = json.loads(data) # this needs to be records orient
    pd_data = pd.read_json(data, orient = "records")
  
    predictions = model.predict(pd_data)
    return json.dumps(predictions.tolist())
    
  except Exception as e:
    result = str(e)
    return json.dumps({"error": result})
  
""", overwrite = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Generate an inference configuration
# MAGIC We use the `InferenceConfig` class to create an inference configuration for deploying the model. It contains the path to the entry script, as well as the environment that we want to use.

# COMMAND ----------

from azureml.core.model import InferenceConfig

# COMMAND ----------

model_inference_config = InferenceConfig(
  environment = env,
  source_directory = "/dbfs/tmp/mthone/",
  entry_script = "model_score.py"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Load our model from MLFlow to AzureML
# MAGIC We get our model from MLFlow model registry (but you can also just get it from a MLFlow experiment), and save it locally. We can then load the `.pkl` file using the `pickle` library, and   add this model to the AzureML model registry.

# COMMAND ----------

from azureml.core.model import Model

# COMMAND ----------

# get model from model registry
modelRegistryName = "mthone_telco_churn"
model_uri = f"models:/{modelRegistryName}/production"
my_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# directory on the driver where we save our model
temp_model_dir = "/dbfs/User/carsten.thone@databricks.com/tmp_model/"
temp_model_path = temp_model_dir + "model.pkl"

# COMMAND ----------

mlflow.sklearn.save_model(my_model, temp_model_dir)

# COMMAND ----------

model = Model.register(
  model_path = temp_model_path,
  model_name = "xgboost_churn_model",
  tags = {"run_id": my_model.metadata.run_id},
  workspace=ws)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Deploy the model to AzureML
# MAGIC We are now ready to create a real-time endpoint! We first create a "deployment configuration" that tells AzureML how many cores/memory to use for this particular web service. We can then do `Model.deploy()` to deploy our AzureML registered model to AzureML.

# COMMAND ----------

from azureml.core.webservice import AksWebservice, Webservice

# COMMAND ----------

deployment_config = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 1, enable_app_insights = True)

service = Model.deploy(
  workspace = ws,
  name = "xgb-inference",
  models = [model],
  inference_config = model_inference_config,
  deployment_config = deployment_config,
  deployment_target = compute_target
)

# COMMAND ----------

bearer_token = service.get_keys()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate predictions!
# MAGIC Now that our service is deployed, we can generate predictions. Just to test that this works, we will load in data from our Feature Store, convert that data to json, and send it to the webservice. In addition, we need to get an API token from the webservice, that we use for Bearer authentication when generating predictions.
# MAGIC 
# MAGIC **Note**: The best practice here is to create an [online store](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/feature-store/concepts#--online-store) of our Feature Store, for low latency predictions. In the future we can query from this online store outside Databricks, so that we do not need Databricks for AzureML inferencing (we'd just use it for training.)

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup
import pandas as pd
fs_client = FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Load the data (we "pretend" it is new inference data)
inference_data = spark.table("telco_db.inference_data").select("customerID", "LastCallEscalated")

# COMMAND ----------

# generate all look ups
def generate_all_lookups(table):
    return [FeatureLookup(table_name = table.name, lookup_key = "customerID", feature_names = f) for f in fs_client.read_table(table.name).columns if f != "customerID"]


def build_training_data(inference_data, feature_tables):
  
  feature_lookups = []
  
  for ft in feature_tables:
    feature_table = fs_client.get_feature_table(ft)
    feature_lookup = generate_all_lookups(feature_table)
    feature_lookups += feature_lookup
    
  return fs_client.create_training_set(inference_data, feature_lookups, label = None, exclude_columns = "customerID")

# COMMAND ----------

# This is the set of customer IDs for which we want to train the data, including our "inference time" features
database_name = "telco_db"
inference_data = spark.table(f"{database_name}.inference_data").select("LastCallEscalated","customerID")

## Training Data will be needed for fs.log_model
training_data = build_training_data(inference_data, [f"{database_name}.service_features",f"{database_name}.demographic_features"])
training_df = training_data.load_df().toPandas()

# Make sure columns are ordered the same as in training
training_df = training_df.reindex(sorted(training_df.columns), axis=1)

# Convert to json, and just take 10 records
json_data = training_df.iloc[0:10].to_json(orient = "records")

# COMMAND ----------

display(training_df)

# COMMAND ----------

json_data

# COMMAND ----------

service.scoring_uri

# COMMAND ----------

import requests
import json

bearer_token = service.get_keys()[0]  # retrieve api tokens for authentication
uri = service.scoring_uri             # retrieve scoring uri

headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {bearer_token}"}

data = json_data

response = requests.post(uri, data=data, headers=headers)

# COMMAND ----------

p = response.json()

# COMMAND ----------

p

# COMMAND ----------


