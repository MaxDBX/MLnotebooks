# Databricks notebook source
#dbutils.widgets.text("modelRegistryName","mthoneBankXGB")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

def getProdModelURI(modelRegistryName):
  models = client.search_model_versions("name='%s'" % modelRegistryName)
  source = [model for model in models if model.current_stage == "Production"][0].source
  return source

# COMMAND ----------

modelURI = getProdModelURI(modelRegistryName)

# COMMAND ----------

dbutils.secrets.get("fieldeng","mthone_sp_pw")

# COMMAND ----------

import os

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


## FILL IN YOUR OWN ACCESS TOKENS BELOW (BEST PRACTICE: MAKE USE OF DBUTILS SECRETS)
svc_pr = ServicePrincipalAuthentication(
    tenant_id=dbutils.secrets.get("fieldeng","mthone_tenant_id"),
    service_principal_id=dbutils.secrets.get("fieldeng","mthone_sp_id"),
    service_principal_password=dbutils.secrets.get("fieldeng","mthone_sp_pw"))

# Use interactive login (base option)
# Otherwise use service principal
ws = Workspace(
    subscription_id=dbutils.secrets.get("fieldeng","mthone_subscription_id"), # Azure subscription ID
    resource_group="mthone-fe", # Name of resource group in which Azure ML workspace is deployed
    workspace_name="mthoneML",  # Name of Azure ML workspace 
    auth=svc_pr
    )
print("Found workspace {} at location {}".format(ws.name, ws.location))

# COMMAND ----------

import mlflow.azureml

# run_id_python = "runs:/e3f4485070bf433cab95836ed22d1fee/model"
run_id_python = modelURI

model_image, azure_model = mlflow.azureml.build_image(workspace=ws, 
                                                      model_uri=run_id_python,
                                                      model_name="xgb_bank_model",
                                                      image_name="xgbbankimg",
                                                      description="xgboost model for Bank Marketing Dataset", 
                                                      tags={
                                                        "tag1": str("bla1"),
                                                        "tag2": str("bla2"),
                                                      },
                                                      synchronous=False)

# COMMAND ----------

model_image.wait_for_creation(show_output=True)

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget

# Get the resource id from https://porta..azure.com -> Find your resource group -> click on the Kubernetes service -> Properties
#resource_id = "/subscriptions/<your subscription id>/resourcegroups/<your resource group>/providers/Microsoft.ContainerService/managedClusters/<your aks service name>"
# Attach the cluster to your workgroup
attach_config = AksCompute.attach_configuration(resource_group = "mthone-fe", # name of resource group in which AKS service is deployed
                                         cluster_name = "mthoneAKS")  # name of AKS service
aks_target = ComputeTarget.attach(ws, 'mthone-ml-aks', attach_config) # To be defined name of the compute target in Azure ML workspace (can be defined here)

# Wait for the operation to complete
aks_target.wait_for_completion(True)
print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)

# COMMAND ----------

from azureml.core.webservice import Webservice, AksWebservice

# Set configuration and service name
prod_webservice_name = "bankxgboost"
prod_webservice_deployment_config = AksWebservice.deploy_configuration()

# Deploy from image
prod_webservice = Webservice.deploy_from_image(workspace = ws, 
                                               name = prod_webservice_name,
                                               image = model_image,
                                               deployment_config = prod_webservice_deployment_config,
                                               deployment_target = aks_target)

# COMMAND ----------

prod_scoring_uri = prod_webservice.scoring_uri
prod_service_key = prod_webservice.get_keys()[0] if len(prod_webservice.get_keys()) > 0 else None

print(prod_scoring_uri)
print(prod_service_key)

# COMMAND ----------

import requests
import json

def query_endpoint_example(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=inputs, headers=headers)
  print(response)
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds

# COMMAND ----------

df = spark.sql("select * from max_db.bank_marketing_train_set")

# COMMAND ----------

train_x = df.toPandas()
sample = train_x.iloc[:,:]
sample_json = sample.to_json(orient="split")

# COMMAND ----------

sample_json

# COMMAND ----------

prod_prediction = query_endpoint_example(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=sample_json)

# COMMAND ----------

prod_prediction

# COMMAND ----------

