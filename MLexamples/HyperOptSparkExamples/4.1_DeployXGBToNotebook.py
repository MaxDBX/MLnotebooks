# Databricks notebook source
# MAGIC %md 
# MAGIC ###Deploying our hyperOpt model in a notebook
# MAGIC In this notebook we show how you can deploy your model in a databricks notebook. We will load our model from the product path in our Registered Model, and subsequently we will wrap it in a Pandas UDF, and get predictions. Since we also want to deploy this notebook as an automated job, we will use [databricks library utilities](https://docs.databricks.com/dev-tools/databricks-utils.html#library-utilities) to install notebook-scoped libraries.

# COMMAND ----------

# MAGIC %md
# MAGIC #### A note on cluster configuration
# MAGIC * For deployment you may want to opt for Databricks Run time, since you can then make use of notebook scoped libraries (this will soon also be available for Machine Learning Runtime)
# MAGIC * **Bug** Currently there is a bug in which `import mlflow` within `pandas_udf` throws a java connection error for ML runtime. As such, to make use of running predictions in pandas_udf, please stick with Databricks runtime (with same version as ML runtime) for now, and install mlflow and xgboost==0.90 on the cluster
# MAGIC   * Make sure xgboost is same version as used for training, which is very likely to be 0.90

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Get model URI associated with our model in Model Registry production stage

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

dbutils.widgets.text("modelRegistryName","bankXGBoost")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

def getProdModelURI(modelRegistryName):
  models = client.search_model_versions("name='%s'" % modelRegistryName)
  source = [model for model in models if model.current_stage == "Production"][0].source
  return source

modelURI = getProdModelURI(modelRegistryName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create a scalar pandas UDF for predictions
# MAGIC In the next step we build a pandas UDF that:
# MAGIC * uses mlflow.xgboost to load the xgboost model using our model URI
# MAGIC * gets predictions by executing xgb_model.predict on our features data

# COMMAND ----------

## Create pandas udf
import pyspark.sql.functions as f
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import FloatType, IntegerType

# Need api_token and api_url in order to enable the worker nodes to connect to MLFlow tracking server
cntx = dbutils.entry_point.getDbutils().notebook().getContext()
api_token = cntx.apiToken().get()
api_url = cntx.apiUrl().get()

# Create a scalar pandas UDF for our predictions.
@pandas_udf(FloatType())
def predict(data):
  import os
  import pandas as pd
  import mlflow
  import mlflow.xgboost
  from xgboost import DMatrix
  
  os.environ["DATABRICKS_TOKEN"] = api_token
  os.environ["DATABRICKS_HOST"] = api_url
  
  # Change path to local (worker nodes do not have access to dbfs:/)
  
  print("Loading Model")
  localModelURI = "/" + modelURI.replace(":","")
  xgb_model = mlflow.xgboost.load_model(localModelURI)
  
  # Predictions
  print("Generating predictions")
  predictions = pd.Series(xgb_model.predict(DMatrix(data)))
  return predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Generating predictions
# MAGIC Now that we have our pandas UDF defined, we are going to load in our bank marketing data set, on which we execute the pandas UDF to get predictions.

# COMMAND ----------

categoricalColumns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
numericCols = ["age", "balance", "campaign", "previous", "day"]

featureColumns = numericCols + [c + "Index" for c in categoricalColumns]

# COMMAND ----------

df = spark.sql("select * from global_temp.globalTempTrainData")

# COMMAND ----------

featureDF = df.select(featureColumns)
import pyspark.sql.functions as f

resultDF = (featureDF
      .withColumn("features",f.struct([f.col(x) for x in (featureColumns)]))
      .withColumn("predictions",predict(f.col("features"))))

# COMMAND ----------

resultDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scheduling the notebook
# MAGIC A logical next step is to schedule this notebook. This is easily done:
# MAGIC * Click on the Jobs button in the left vertical bar, and subsequently click on "create job".
# MAGIC * Subsequently: 
# MAGIC   * In task, select notebook and select this notebook.
# MAGIC   * In parameters, add the widget name (modelRegistryName) as key name and the registered model as value (bankXGBoost)
# MAGIC * For cluster: Make sure to always select a new cluster.
# MAGIC * In schedule you can then select how often you want to run the job.
# MAGIC * with respect to libraries: Make sure to install them in the notebook using dbutils.library.installPyPI()