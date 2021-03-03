# Databricks notebook source
# MAGIC %md
# MAGIC ### Cluster Configuration
# MAGIC Make sure to start a cluster with the following configurations:
# MAGIC * Databricks Runtime Version: 7.0+
# MAGIC * Instance type: Standard_F8s (5 nodes will be more than enough)
# MAGIC   * We will be using pandas UDF to build many models, and then compute-optimised instance types are preferred.
# MAGIC * Libraries:
# MAGIC   * azureml-sdk[databricks] (for deploying to AzureML)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Introduction
# MAGIC In this set of three notebooks we will demonstrate the following:
# MAGIC * Building a single xgboost model
# MAGIC * Building multiple xgboost models in parallel by making use of pandas_UDF
# MAGIC * Use MLFlow Tracking UI to: 
# MAGIC   * save metrics of all ML runs
# MAGIC   * log the generated XGBoost models.
# MAGIC * Use MLFlow Model Registry to log our best model in a specified "production path"
# MAGIC * Use Mlflow Azure functionality to deploy our model to AWS Sagemaker.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook 1: Training many XGBoost models
# MAGIC In this notebook we will use a data-set about direct marketing campaigns of a Portuguese Banking Institution to predict whether someone will subscribe a term deposit (variable y). This data-set has been downloaded and put into a delta table in notebook 00_preparedata. Also see: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing. In order to do this we take the following steps:
# MAGIC 1. Perform basic feature engineering to make the data readable for our XGBoost model.
# MAGIC 2. Create an MLFlow experiment to track our XGBoost training runs, and to log our XGBoost models.
# MAGIC 3. Create multiple XGBoost models for various hyperparameters in parallel using Pandas UDF.
# MAGIC 4. Find out which is the best XGBoost model by looking at the MLFlow UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run set up

# COMMAND ----------

# DBTITLE 0,Run set up
# MAGIC %run /Projects/carsten.thone@databricks.com/MLnotebooks/MLexamples/includes/mlSetup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get the data

# COMMAND ----------

dataset = table("bank_db.bank_marketing_train_set")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create hyperparameter grid
# MAGIC In the cells below:
# MAGIC * We create a spark dataframe that contains a row for each combination of hyperparameters we'd like to evaluate.
# MAGIC * We then cross join this to the dataset, so that we have a dataset for each row of hyperparameters. This is a convenient way to pass the dataset to Pandas UDF for each set of hyper parameters.

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql import Row

xgb_grid = [{"n_estimators":[200,300,350,400,450,500,600,700],
           "max_depth":[3,4,5,6,7]}]

spark_grid = spark.createDataFrame(Row(**x) for x in xgb_grid)

exploded_grid = (
  spark_grid
  .withColumn('n_estimators',f.explode(f.col('n_estimators')))
  .withColumn('max_depth',f.explode(f.col('max_depth'))))

# COMMAND ----------

display(exploded_grid)

# COMMAND ----------

cross_val_set = dataset.crossJoin(exploded_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create and set MLFlow experiment

# COMMAND ----------

# MAGIC %md
# MAGIC In the below cell we are creating a new MLFlow experiment.
# MAGIC * experiment_path denotes the path in your workspace (see Workspace in the vertical icon bar on the left) where the MLFlow experiment is created
# MAGIC * mlflow_model_save_dir will be the DBFS (=ADFS) path that all the artifacts related to the experiment are stored (most notably the model objects)
# MAGIC * **make sure to fill in your ws_user with your email that's used to login to databricks (and which is the name of your home folder)**

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri("databricks") # if databricks -> then 'MANAGED' and somewhere on the control plane
ws_user = "carsten.thone@databricks.com"
experiment_path = "/Users/{}/2.manyXGBoostModels".format(ws_user)

# COMMAND ----------

experiment_id = setOrCeateMLFlowExperiment(experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run pandas UDF to create multiple xgboost models in parallel

# COMMAND ----------

# MAGIC %md
# MAGIC Below we create a Pandas UDF to run multiple xgboost models on worker nodes in parallel. Since we require the worker nodes to log MLFlow parameters to the MLFlow tracking server, we need to set the `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables in the worker nodes. This is done in two lines below importing the libraries  

# COMMAND ----------

## Create pandas udf
from pyspark.sql.types import DoubleType, StructType, StructField, StringType

# MLFlow in pandas udf
cntx = dbutils.entry_point.getDbutils().notebook().getContext()
api_token = cntx.apiToken().get()
api_url = cntx.apiUrl().get()

schema = StructType([
  StructField('run_id', StringType(), True),
  StructField('n_estimators',DoubleType(),True),
  StructField('max_depth',DoubleType(),True),
  StructField('auc', DoubleType(), True)])

def train_xgboost(data):
  print("STARTING EXECUTION")
  # Import libraries
  import os
  import pandas as pd
  from sklearn.metrics import roc_auc_score
  from sklearn.model_selection import train_test_split
  from xgboost import XGBClassifier
  import mlflow
  import mlflow.xgboost
  
  # Set internal token and url to communicate with tracking server
  os.environ["DATABRICKS_TOKEN"] = api_token
  os.environ["DATABRICKS_HOST"] = api_url
  
  # retrieve hyper parameters and drop them from the dataframe
  n_estimators = data['n_estimators'].iloc[0]
  max_depth = data['max_depth'].iloc[0]
  
  print("building model for %d estimators and %d depth" %(n_estimators, max_depth))
  data = data.drop(["n_estimators", "max_depth"], axis=1)
  
  # Split into train and test data, and features and target variable
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
  predictions = pd.DataFrame(xgb_model.predict(test_x), columns=["Predictions"])
  auc = roc_auc_score(test_y, predictions['Predictions'])
  
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

spark.conf.set("spark.sql.shuffle.partitions",sc.defaultParallelism * 2)

# COMMAND ----------

result_set = cross_val_set.groupby("n_estimators","max_depth").applyInPandas(train_xgboost, schema = schema)

# COMMAND ----------

display(result_set)

# COMMAND ----------

# We can also use the mlflow client to retrieve our best run_id
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

best_run_id = client.search_runs(experiment_ids = [experiment_id], order_by=["metrics.auc DESC"])[0].info.run_id

# COMMAND ----------

best_run_id

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add model to Model Registry
# MAGIC We can now add our model to Model Registry. Included is a notebook that we can use to retrieve a model from the MLFlow experiment, and subsequently add it to model registry.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Appendix: alternative to grid-search
# MAGIC In this notebook we did a simple grid-search to find a model with the best hyperparameters. However, Databricks also supports more advanced methods, such as Hyperopt. See [this link](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/automl/hyperopt/) for more information.

# COMMAND ----------


