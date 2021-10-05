# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

dbutils.widgets.text("modelRegistryName","FILL_IN")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying our model to a Databricks notebook
# MAGIC Now that we pushed a model to the production stage in our model registry, we can deploy it from there to a Databricks notebook. We can now again use the Feature Store to easily do this. Remember that the Feature Store has feature computations associated with this. This means we make sure we use the same computations for training and inference, which reduces the chance of weird inconsistencies!

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Load our model from the model registry
# MAGIC We get the model registry name, and stage, and can then simply construct a model uri as `model_uri = models:/<model_registry_name>/<stage>`. Now that we use [`mlflow.pyfunc.spark_udf`](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf). This enables us to get predictions from our model using a Spark UDF. In other words, when we generate predictions, this is done in a parallel fashion, using the entire cluster. This leads to a dramatic speed up compared to generating predictions from just the driver node (which is how you would do it on your local machine).

# COMMAND ----------

import mlflow
model_uri = f"models:/{modelRegistryName}/production"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Load some data and generate predictions!
# MAGIC Now that we have loaded our model in a spark udf, we can use it to generate predictions. For this example, we simply just re-use our training data, but this would of course also work with any "new" data. Because our model is associated with the Feature Store, all we have to do is supply it the `customerID`, and the "last minute feature" `LastCallEscalated`. The `FeatureStoreClient` will join up these `customerID` with the `delta tables` associated with the FeatureStore, and retrieve all the associated features. This makes batch scoring a lot easier, but more importantly, **it ensures consistency between computations at training time and computations at inferencing time**

# COMMAND ----------

# This is just the train data we used before but we "pretend" it's new data just to show how this works.
inference_data = spark.table("max_db.inference_data").select("customerID", "LastCallEscalated")

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# COMMAND ----------



# COMMAND ----------

with_predictions = fs.score_batch(f"models:/{modelRegistryName}/production", inference_data, result_type = "string")

# COMMAND ----------

display(with_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Online scoring with Rest API
# MAGIC The Feature Store also has an online store. Essentially this is a AWS RDS that makes a copy of these features in the Delta Tables, which allows for lower latency model serving. However at this point in time Serving a model using the Online Store is not supported yet. Currently you must load the data from the feature store, or from delta table, and convert it to json before creating a REST API scoring request. for more details see [here](https://docs.databricks.com/applications/mlflow/model-serving.html#score-via-rest-api-request)

# COMMAND ----------


