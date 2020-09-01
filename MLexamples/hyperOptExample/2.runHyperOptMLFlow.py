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
experiment_path = "/Users/{}/mlflowExperiments/RF".format(username) # workspace path by default
#mlflow_model_save_dir = "/Users/{}/mlflowExperiments/hyperOptExample".format(ws_user) # dbfs path (i.e. path to your root bucket, or some mounted ADFS folder)
mlflow_model_save_dir = "{}/mlflowExperiments/RF".format(userhome)

# COMMAND ----------

def setOrCeateMLFlowExperiment(experimentPath,mlflow_model_save_dir):
  from mlflow.exceptions import MlflowException
  try:
    experiment_id = mlflow.create_experiment(experimentPath, "dbfs:" + mlflow_model_save_dir)
  except MlflowException: # if experiment is already created
    experiment_id = mlflow.get_experiment_by_name(experimentPath).experiment_id
    mlflow.set_experiment(experimentPath)
  return experiment_id

# COMMAND ----------

experiment_id = setOrCeateMLFlowExperiment(experiment_path,mlflow_model_save_dir)

# COMMAND ----------

def trainNotebook(params):
  numEstimators= str(params["numEstimators"])
  maxDepth = str(params["maxDepth"])
  
  # can add parameter here, for example: tweedieVariancePAower = params["tweedie_variance_power"]
  # Make sure to add it to the widget of the notebook you want to run, and to the arguments in dbutils.notebook.run
  
  print(numEstimators)
  print(maxDepth)
  
  with mlflow.start_run(experiment_id = experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_param("numEstimators", numEstimators)
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("modelType","XGB")
    str_loss = dbutils.notebook.run("models/runXGBoost4J", timeout_seconds = 600, arguments = {"max_depth": maxDepth, 
                                                                                              "n_estimators": numEstimators, 
                                                                                              "experiment_id": experiment_id,
                                                                                              "run_id": run_id
                                                                                             })
    print(str_loss)
    mlflow.log_metric("loss", float(str_loss))
  return {"loss": float(str_loss), "status": STATUS_OK}

# COMMAND ----------

# Run HyperOpt
best_param = fmin(
  fn=trainNotebook,
  space=space,
  algo=algo,
  max_evals=2,
  return_argmin=False,
)

# COMMAND ----------

# Test RF model
import mlflow
import mlflow.spark

mlflow.spark.load_model("runs:/ff72286706ce4059bb9885587d28fce9/model")

# COMMAND ----------

