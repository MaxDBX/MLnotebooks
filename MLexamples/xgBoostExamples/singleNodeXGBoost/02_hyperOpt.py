# Databricks notebook source
# MAGIC %md
# MAGIC ### Training a single-node model with HyperOpt, using features from the Feature Store
# MAGIC 
# MAGIC Next, it’s time to build a simple model predicting whether the customer churns. There are two feature store tables containing information we want to join to the other ‘inference data’ in order to build a model:
# MAGIC 
# MAGIC * `demographic_features`
# MAGIC * `service_features`
# MAGIC 
# MAGIC We could of course simply read the data from the tables as a DataFrame and join them. However, in a feature store, we want to define the joins declaratively, because the same join will need to happen in a quite different context in online serving (this is not supported yet).

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Import Libraries and set up MLFlow experiment

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup
from xgboost import XGBClassifier
import mlflow
from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient()
fs_client = FeatureStoreClient()
database_name = "telco_db"

# COMMAND ----------

# MAGIC %run /Repos/carsten.thone@databricks.com/MLnotebooks/MLexamples/includes/mlSetup

# COMMAND ----------

experiment_path = "/Users/{}/1.churn_example".format(username) # Workspace path
experiment_id = setOrCeateMLFlowExperiment(experiment_path)

experiment = mlflow_client.get_experiment(experiment_id)
mlflow.set_experiment(experiment.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature Lookups
# MAGIC Below, a FeatureLookup is created for each feature in the service_features table. This defines how other features are looked up (table, feature names) and based on what key – customerID. `create_training_set` pulls together all the features. Its “input” is the `inference_data` table, as discussed above, and the FeatureLookups define how to join in features. Modeling ignores the `customerID` of course; it’s a key rather than useful data.

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
    
  return fs_client.create_training_set(inference_data, feature_lookups, label = "Churn", exclude_columns = "customerID")

# COMMAND ----------

# This is the set of customer IDs for which we want to train the data, including our "inference time" features
inference_data = spark.table(f"{database_name}.inference_data")

# COMMAND ----------

# This is the set of customer IDs for which we want to train the data, including our "inference time" features
inference_data = spark.table(f"{database_name}.inference_data")

## Training Data will be needed for fs.log_model
training_data = build_training_data(inference_data, [f"{database_name}.service_features",f"{database_name}.demographic_features"])
training_df = training_data.load_df().toPandas()

# COMMAND ----------

display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Single Node Model training with Hyper Opt
# MAGIC We are now going to create an sklearn Pipeline Model for our XGBoost and train this with HyperOpt. A question that may arise is: What part of "feature engineering" should be in the "sklearn pipeline" model, and what part of feature engineering should go before the FeatureStore?
# MAGIC 
# MAGIC As an example, consider XGBoost. XGBoost does not take any string data, so we will need to use label encoders from SKLearn to convert these columns to integers. This is not something you'd want to do before a feature store, since then you would end up with features that have meaningless values, and on which you can not perform any meaningful analytics. We use label encoders specifically for the XGBoost model, so as such these features should only be generated as part of the model training/inferencing.

# COMMAND ----------

cntx = dbutils.entry_point.getDbutils().notebook().getContext()
api_token = cntx.apiToken().get()
api_url = cntx.apiUrl().get()

# COMMAND ----------

# data is a pandas dataframe
def train_model(num_estimators, max_depth, threshold, experiment_id):
  from sklearn.compose import ColumnTransformer
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.ensemble import GradientBoostingClassifier
  from sklearn.metrics import f1_score, roc_auc_score, log_loss
  
  import os
  import pandas as pd
  import mlflow
  import mlflow.sklearn
  
  # Set internal token and url to communicate with tracking server
  os.environ["DATABRICKS_TOKEN"] = api_token
  os.environ["DATABRICKS_HOST"] = api_url
  
  
  # Build the training set
  data = broadcast_data.value
  
  # split into feature and label set
  X = data.drop("Churn", axis = 1)
  y = data["Churn"]
  
  # split into train and test set
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
  
  # Constructing sample weights (alternatively you could downsample the training dataset)
  churn_weight = 1 - (y_train.sum()/ len(y))
  not_churn_weight = y_train.sum()/len(y)
  sample_weight = y_train.map(lambda churn: churn_weight if churn else not_churn_weight)
  
  # One hot encoding on all the categorical variables (note that one hot encoding is not necessarily great for XGBoost)
  encoders = ColumnTransformer(transformers =[("encoder", OneHotEncoder(handle_unknown='ignore'), X.columns[X.dtypes == "object"])])
  
  xgb_classifier = XGBClassifier(max_depth = max_depth, n_estimators = num_estimators)
  pipeline = Pipeline([("encoder", encoders), ("xgb_classifier", xgb_classifier)])
  pipeline_model = pipeline.fit(X_train, y_train, xgb_classifier__sample_weight=sample_weight)
  
  # Get probabilities of test data
  y_proba = pipeline_model.predict_proba(X_test)[:,1]
  # Get predictions based on threshold
  y_pred = y_proba >= threshold
  # Calculate F1 score
  F1 = f1_score(y_test,y_pred)
  auc = roc_auc_score(y_test, y_pred)
  
  with mlflow.start_run(experiment_id = experiment_id) as run:
    mlflow.log_param("num_estimators",num_estimators)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("threshold_1", threshold)
    
    mlflow.log_metric("auc",auc)
    mlflow.log_metric("F1", F1)
    mlflow.sklearn.log_model(pipeline_model, "model")
    
  return 1 - F1

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Using HyperOpt
# MAGIC Hyperopt is a Python library that can optimize a function’s value over complex spaces of inputs. For machine learning specifically, this means it can optimize a model’s accuracy (loss, really) over a space of hyperparameters. It’s a Bayesian optimizer, meaning it is not merely randomly searching or searching a grid, but intelligently learning which combinations of values work well as it goes, and focusing the search there. More information on HyperOpt best practices can be found [here](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) .

# COMMAND ----------

# Single machine HyperOpt /w distributed ML.
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import numpy as np

# COMMAND ----------

#  We want to make use of Bayesian Optimisation algorithm
algo=tpe.suggest

# Space over which we want to evaluate our model
space = {
  "threshold": hp.quniform("threshold",0.5, 0.9, 0.1),
  'numEstimators': hp.quniform('numEstimators', 100, 500, 1),
  'maxDepth': hp.quniform('maxDepth', 3,13, 1)
  # can easily add extra parameters here, for example 'tweedie_variance_power'
}

# COMMAND ----------

# Wrapper function that will be used in fmin
def hyper_opt_wrapper_func(params):
  
  numEstimators = int(params["numEstimators"])
  maxDepth = int(params["maxDepth"])
  threshold = int(params["threshold"] * 20)/20
  
  print(f"trials with numEstimators: {numEstimators}, maxDepth: {maxDepth}, threshold: {threshold}")
  
  loss = train_model(numEstimators, maxDepth, threshold, experiment_id = experiment_id)
  return {"loss": loss, "status": STATUS_OK}

# COMMAND ----------

broadcast_data = sc.broadcast(training_df)

# COMMAND ----------

# Run HyperOpt
best_param = fmin(
  fn=hyper_opt_wrapper_func,
  space=space,
  algo=algo,
  max_evals=16,
  trials = SparkTrials(parallelism = 2),
  return_argmin=False
)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

best_run_id = mlflow_client.search_runs(experiment_ids = [experiment_id], order_by=["metrics.F1 DESC"])[0].info.run_id

# COMMAND ----------

best_run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Associating our best model with the Feature Store
# MAGIC We need to associate our model with the Feature Store, so that we can use the Feature Store compute logic (defined in the last notebook) for inferencing in production. Ideally we'd have done this directly in the hyperOpt function, but the Feature Store is not supported on worker nodes. As a work-around, we have to load the model, and subsequently log it to our model registry using `fs.log_model`. Unfortunately this generates a 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Adding our "best model" to the Model Registry
# MAGIC We can now add our model to the Model Registry. To this end we use `mlflow.register_model`. `register_model` is a function that's part of the `mlflow library`, and it's not a method that's part of the MLflow client.

# COMMAND ----------

run_uri = f"runs:/{best_run_id}/model"
model_registry_name = "max_telco_churn_model"
best_model = mlflow.pyfunc.load_model(run_uri)
mlflow.end_run()

# We want to log the model to the run ID associated with producing this model in hyper opt.
with mlflow.start_run(run_id = best_run_id, experiment_id = experiment_id) as run:
  fs_client.log_model(
    best_model,
    "best_hyperopt_model",
    flavor = mlflow.sklearn,
    training_set = training_data,
    registered_model_name = model_registry_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Adding the model to the Staging stage
# MAGIC Secondly, we can add the the model to the "staging stage". At this stage it's recommended to have a manuel check to make sure the model is correctly tuned, before we trigger the notebook that deploys the model to production.

# COMMAND ----------

modelVersion = mlflow_client.search_model_versions("run_id='%s'" % best_run_id)[0].version

mlflow_client.transition_model_version_stage(
    name=model_registry_name,
    version=modelVersion,
    stage="staging"
)

# COMMAND ----------

best_run_id

# COMMAND ----------


