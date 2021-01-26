# Databricks notebook source
input_data = table('bank_db.bank_marketing_train_set')
cols = input_data.columns

# COMMAND ----------

# Create mlflow experiment
import mlflow
ws_user = "carsten.thone@databricks.com" # fill in your home folder (which is your user email used to login to Azure Databricks)
mlflow.set_tracking_uri("databricks") # if databricks -> then 'MANAGED' and somewhere on the control plane

experiment_path = "/Users/{}/singleNodeMLFlow".format(ws_user)

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

def train_xgboost(data):
  print("STARTING EXECUTION")
  
  import os
  import pandas as pd
  from sklearn.metrics import roc_auc_score
  from sklearn.model_selection import train_test_split
  from xgboost import XGBClassifier
  import mlflow
  import mlflow.xgboost
  
  n_estimators = 250
  max_depth = 5
  print("building model for %d estimators and %d depth" %(n_estimators, max_depth))
  train, test = train_test_split(data)
  
  train_x = train.drop(["label"], axis=1)
  test_x = test.drop(["label"], axis=1)
  train_y = train[["label"]]
  test_y = test[["label"]]
  
  # Train and save an XGBoost model
  print("building model for %d estimators and %d depth" %(n_estimators, max_depth))
  xgbCL = XGBClassifier(max_depth = max_depth, 
                            n_estimators = n_estimators)
  
  print("Fitting model for %d estimators and %d depth" % (n_estimators, max_depth))
  xgb_model = xgbCL.fit(train_x,train_y)
  
  predicted_qualities = pd.DataFrame(xgb_model.predict(test_x), columns=["Predictions"])
  predicted_qualities['probability_0'] = xgb_model.predict_proba(test_x)[:,0]
  predicted_qualities['probability_1'] = xgb_model.predict_proba(test_x)[:,1]
  
  auc = roc_auc_score(test_y, predicted_qualities['Predictions'])
  
  print("Starting MLFLow run for %d estimators and %d depth" %(n_estimators, max_depth))
  with mlflow.start_run(experiment_id = experiment_id) as run:
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("max_depth",max_depth)
    
    mlflow.log_metric("auc",auc)
    mlflow.xgboost.log_model(xgb_model,"model")
    
  predicted_qualities['n_estimators'] = n_estimators
  predicted_qualities['max_depth'] = max_depth
  predicted_qualities['auc'] = auc
  
  return predicted_qualities

# COMMAND ----------

pdDF = input_data.toPandas()

# COMMAND ----------

returnDF = train_xgboost(pdDF)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

best_run_id = client.search_runs(experiment_ids = [experiment_id], order_by=["metrics.auc DESC"])[0].info.run_id

# COMMAND ----------

import mlflow.xgboost
model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/model")

# COMMAND ----------

