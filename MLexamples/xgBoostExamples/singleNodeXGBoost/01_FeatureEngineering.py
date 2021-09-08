# Databricks notebook source
# MAGIC %md
# MAGIC ### 01: Churn Prediction: Feature engineering
# MAGIC 
# MAGIC This simple example will build a feature store on top of data from a telco customer churn data set: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv and then use it to train a model and deploy both the model and features to production.
# MAGIC 
# MAGIC The goal is to produce a service that can predict whether a customer churns.

# COMMAND ----------

# Artificially add a primary key
from databricks.feature_store import FeatureStoreClient
import pyspark.sql.functions as F
fs = FeatureStoreClient()
database_name = 'max_db'
bronze_table_name = "bronze_telco_churn"

input_data = table(f"{database_name}.{bronze_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining Feature Store Tables
# MAGIC 
# MAGIC The raw data set `input_data` can be improved for ML and data science by:
# MAGIC * splitting into logical

# COMMAND ----------

from databricks.feature_store import feature_table
import pyspark.sql.functions as F

demographic_cols = ["customerID", "gender", "seniorCitizen", "partner", "dependents"]
service_cols = ["customerID"] + [c for c in input_data.columns if c not in ["churnString"] + demographic_cols]

# COMMAND ----------

@feature_table
def compute_demographic_features(data):
  import pyspark.sql.functions as F
  
  demographic_cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents"]
  data = data.withColumn("SeniorCitizen", F.col("SeniorCitizen") == F.lit(1))
  
  # Convert Yes/No to Boolean
  for yes_no_col in ["Partner", "Dependents"]:
    data = data.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
  
  return data.select(demographic_cols)

# COMMAND ----------

demographics_df = compute_demographic_features(input_data)

# COMMAND ----------

display(demographics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature table 2: Service Features

# COMMAND ----------

@feature_table
def compute_service_features(data):
  for yes_no_col in ["PhoneService", "PaperlessBilling"]:
    data = data.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
    
    
  # convert month-to-month
  data = (data.withColumn("Contract",
                          F.when(F.col("Contract") == "Month-to-month", 1).
                          when(F.col("Contract") == "One year", 12).
                          when(F.col("Contract") == "Two year", 24)))
  
  return data.select(service_cols)

# COMMAND ----------

service_df = compute_service_features(input_data)

# COMMAND ----------

display(service_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Inference Data
# MAGIC Note that the Feature Store is for features that can be *pre-computed*. However in the real world this is of course not always the case. Some features can only be computed at *inference time*. We are going to create a made up feature that demonstrates this called `lastCallEscalated`.
# MAGIC 
# MAGIC To summarise, Feature Store tables should be seen as "feature look up tables" for features that can be pre-computed (e.g. "past 3 month spend"), or are unlikely to change on a primary key (e.g. "gender")

# COMMAND ----------

import pyspark.sql.functions as F

def build_inference_df(data):
  
  inference_df = (data.select("customerID", F.col("churnString").alias("Churn"))
                  .withColumn("Churn", F.col("Churn") == "Yes")
                  .withColumn("LastCallEscalated",
                              F.when(F.col("Churn"), F.hash(F.col("customerID")) % 100 < 35)
                              .otherwise(F.hash(F.col("customerID")) % 100 < 15)))

  
  return inference_df

# COMMAND ----------

inference_df = build_inference_df(input_data)

# COMMAND ----------

display(inference_df)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS max_db.service_features")
spark.sql("DROP TABLE IF EXISTS max_db.demographic_features")
spark.sql("DROP TABLE IF EXISTS max_db.inference_data")

# COMMAND ----------

demographic_features_table = fs.create_feature_table(
  name = "max_db.demographic_features",
  keys = "customerID",
  schema = demographics_df.schema,
  description = "Telco customer demographics")

# COMMAND ----------

service_features_table = fs.create_feature_table(
  name = "max_db.service_features",
  keys = "customerID",
  schema = service_df.schema,
  description = "Telco service features")

# COMMAND ----------

compute_demographic_features.compute_and_write(input_data, feature_table_name = "max_db.demographic_features", mode = "merge")

# COMMAND ----------

compute_service_features.compute_and_write(input_data, feature_table_name = "max_db.service_features", mode = "merge")

# COMMAND ----------

inference_df.write.format("delta").saveAsTable("max_db.inference_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Adding more features
# MAGIC Consider the situation where we already trained a few models, and we find that we should add a bunch of features to our tables. How would we do this?

# COMMAND ----------

def buildFeatureData(input_data):
  from pyspark.ml import Pipeline
  from pyspark.ml.feature import StringIndexer
  
  # define names/prefixes for the output columns
  label_col = "label"
  id_col = "customerID"
  
  catInputCols = ["gender","seniorCitizen","partner","dependents", "phoneService","multipleLines",
                  "internetService","onlineSecurity","onlineBackup","deviceProtection","techSupport",
                  "streamingTV","streamingMovies","contract","paperlessBilling","paymentMethod"]
  catOutputCols = [col + "_idx" for col in catInputCols]
  
  print(catInputCols)
  print(catOutputCols)
  
  
  # pipeline stages (Converting string columns to indexed columns)
  string_indexer =  StringIndexer(inputCols = catInputCols, outputCols = catOutputCols)
  label_indexer = StringIndexer(inputCol = "churnString", outputCol = "label")
  stages = [string_indexer, label_indexer]
  pipeline = Pipeline(stages=stages)
  
  # Note: ONLY feature transformations that apply to BOTH train and test set should go here
  # Transformations that apply only to train set, and should be "fitted" to test set should be part of the model training itself
  # (For example: imputing null values of a column with its mean)

  pipelineModel = pipeline.fit(input_data)
  output_data = pipelineModel.transform(input_data)
  numericCols = ["tenure","monthlyCharges","totalCharges"]
    
  return output_data.select([id_col] + numericCols + catOutputCols + [label_col])

# COMMAND ----------

feature_data = buildFeatureData(input_data)

# COMMAND ----------

display(feature_data)

# COMMAND ----------

feature_schema = feature_data.schema

# COMMAND ----------

feature_schema

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store
# MAGIC Now that we have created our feature data set, we will store it into a feature store. Essentially the feature store is a delta-backed table with a number of extra features that work nicely with our features.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Feature Table (only have to do this once)

# COMMAND ----------

from databricks.feature_store import feature_table, FeatureStoreClient
fs = FeatureStoreClient()

# COMMAND ----------

telco_churn_feature_table = fs.create_feature_table(
  name = "max_db.churn_features",
  keys = "customerID",
  schema = feature_data.schema,
  description = "Telco Churn Features"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Write to Feature Table
# MAGIC Essentially we now have an empty Delta Table. We are now going to write our feature engineering data (`feature_data`) to this table. It works pretty much the same as a normal Delta Table

# COMMAND ----------

fs.write_table(
  name = "max_db.churn_features",
  df = feature_data,
  mode = "merge"
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Why use a Feature Store table instead of a normal Delta Table? What is the addtional benefit?
# MAGIC * Easy out of the box addition of data, based on primary keys (e.g. customer IDs in this case.)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note: Edge cases
# MAGIC 
# MAGIC * How would the feature store take care of imputation?
