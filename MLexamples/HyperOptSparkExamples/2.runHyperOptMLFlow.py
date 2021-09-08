# Databricks notebook source
# Single machine HyperOpt /w distributed ML.
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np

#  We want to make use of Bayesian Optimisation algorithm
algo=tpe.suggest

# Space over which we want to evaluate our model
space = {
  'numEstimators': hp.choice('numEstimators', np.arange(100,500, dtype = int)),
  'maxDepth': hp.choice('maxDepth', np.arange(3, 7, dtype=int))
  # can easily add extra parameters here, for example 'tweedie_variance_power'
}

# COMMAND ----------

# MAGIC %run ../includes/mlSetup

# COMMAND ----------

import mlflow
ws_user = "carsten.thone@databricks.com" # fill in your home folder (which is your user email used to login to Azure Databricks)
mlflow.set_tracking_uri("databricks") # if databricks -> then 'MANAGED' and somewhere on the control plane
experiment_path = "/Users/{}/xgBoost4j".format(username) # workspace path by default
#mlflow_model_save_dir = "/Users/{}/mlflowExperiments/hyperOptExample".format(ws_user) # dbfs path (i.e. path to your root bucket, or some mounted ADFS folder)

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

experiment_id = setOrCeateMLFlowExperiment(experiment_path)

# COMMAND ----------

def trainNotebook(params):
  numEstimators= str(params["numEstimators"])
  maxDepth = str(params["maxDepth"])
  
  with mlflow.start_run(experiment_id = experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_param("numEstimators", numEstimators)
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("modelType","XGB")
    str_loss = dbutils.notebook.run("models/runXGBoost4J", timeout_seconds = 600, 
                                    arguments = {"max_depth": maxDepth, 
                                                 "n_estimators": numEstimators, 
                                                 "experiment_id": experiment_id,
                                                 "run_id": run_id})
    mlflow.log_metric("loss", float(str_loss))
  return {"loss": float(str_loss), "status": STATUS_OK}

# COMMAND ----------

# Run HyperOpt

best_param = fmin(
  fn=trainNotebook,
  space=space,
  algo=algo,
  max_evals=4,
  return_argmin=False,
)

# COMMAND ----------

import mlflow
loaded_model = mlflow.pyfunc.spark_udf(spark, "runs:/6d515e2f84ac4d5ea9ad62a5e6e44b1a/model")
