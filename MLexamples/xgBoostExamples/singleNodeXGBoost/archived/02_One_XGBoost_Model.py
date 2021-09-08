# Databricks notebook source
# MAGIC %run /Projects/carsten.thone@databricks.com/MLnotebooks/MLexamples/includes/mlSetup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment tracking example
# MAGIC Below is an example in which we track the training of a simple XGBoost model. We do the following:
# MAGIC * We create a MLflow experiment using `mlflow.create_experiment`, which creates an experiment object for us in the workspace that we can interact with.
# MAGIC * We train a XGBoost model, and log the hyperparameters, metrics and the model object itself to the experiment.

# COMMAND ----------

# Create mlflow experiment
import mlflow
mlflow.set_tracking_uri("databricks")
experiment_path = "/Users/{}/1.singleNodeMLFlow".format(username)

# uses mlflow.create_experiment(experimentPath)
experiment_id = setOrCeateMLFlowExperiment(experiment_path)

# COMMAND ----------

input_data = table('bank_db.bank_marketing_train_set')
pdDF = input_data.toPandas()

# COMMAND ----------

pdDF

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
  auc = roc_auc_score(test_y, predicted_qualities['Predictions'])
  
  print("Starting MLFLow run for %d estimators and %d depth" %(n_estimators, max_depth))
  with mlflow.start_run(experiment_id = experiment_id) as run:
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("max_depth",max_depth)
    
    mlflow.log_metric("auc",auc)
    mlflow.xgboost.log_model(xgb_model,"model")
    
    output_df = pd.DataFrame(data = {"run_id": [run.info.run_uuid], 
                                     "n_estimators": n_estimators,
                                     "max_depth": max_depth,
                                     "auc": [auc]})
  
  return output_df

# COMMAND ----------

input_data = table('bank_db.bank_marketing_train_set')
pdDF = input_data.toPandas()

# COMMAND ----------

returnDF = train_xgboost(pdDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Extra: Use the MLflow client to analyse the runs programmatically
# MAGIC Next to using the Experiments UI, or just putting all the results from a run into spark DF, the `MLflowClient` class can be used to look up runs programmatially from a given (list of) experiment(s). Below is an example of how we can retrieve the run_id with the highest area under the curve score by using the `client.search_runs` method.

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
best_run_id = client.search_runs(experiment_ids = [experiment_id], order_by=["metrics.auc DESC"])[0].info.run_id

# COMMAND ----------

best_run_id

# COMMAND ----------


