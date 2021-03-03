# Databricks notebook source
dbutils.widgets.text("modelRegistryName","mthoneSingleNodeXGB")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying our model to a Databricks notebook
# MAGIC Now that we pushed a model to the production stage in our model registry, we can deploy it from there to a Databricks notebook. This notebook can then subsequently be scheduled using the jobs UI on the left.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Load our model from the model registry
# MAGIC We get the model registry name, and stage, and can then simply construct a model uri as `model_uri = models:/<model_registry_name>/<stage>`. Now that we use [`mlflow.pyfunc.spark_udf`](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf). This enables us to get predictions from our model using a Spark UDF. In other words, when we generate predictions, this is done in a parallel fashion, using the entire cluster. This leads to a dramatic speed up compared to generating predictions from just the driver node (which is how you would do it on your local machine).

# COMMAND ----------

import mlflow

model_name = modelRegistryName
stage = "Production"


loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{stage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Load some data and generate predictions!
# MAGIC Now that we have loaded our model in a spark udf, we can use it to generate predictions. For this example, we simply just re-use our training data, but this would of course also work with any "new" data.

# COMMAND ----------

# This is just the train data we used before but we "pretend" it's new data just to show how this works.
features = table('bank_db.bank_marketing_train_set').drop("label")
feature_cols = features.columns

# Load model as a Spark UDF.
predictions = features.withColumn("prediction",loaded_model(*feature_cols))

# COMMAND ----------

display(predictions)
