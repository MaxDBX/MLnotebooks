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
database_name = 'telco_db'
bronze_table_name = "bronze_telco_churn"

input_data = table(f"{database_name}.{bronze_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining Feature Store Tables
# MAGIC 
# MAGIC The raw data set `input_data` can be improved for ML and data science by:
# MAGIC * splitting into logical, reusable subsets of columns
# MAGIC * engineering more useful features
# MAGIC * Publish result as feature store tables
# MAGIC 
# MAGIC First we define a `@feature_table` that simply selects some demographic information from the data. This will become one feature store table when written later. A `@feature_table` is really just a function that computes a DataFrame defining the features in the table, from a source ‘raw’ DataFrame. It can be called directly for testing; this by itself does not persist or publish features.

# COMMAND ----------

display(input_data)

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
# MAGIC Next, feature related to the customer’s telco service are likewise selected. Note that each of these tables includes customerID as a key for joining. We do some very basic feature engineering here, e.g. converting Contract to number of months.

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

spark.sql(f"DROP TABLE IF EXISTS {database_name}.service_features")
spark.sql(f"DROP TABLE IF EXISTS {database_name}.demographic_features")
spark.sql(f"DROP TABLE IF EXISTS {database_name}.inference_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Writing Feature Store Tables
# MAGIC 
# MAGIC With the tables defined by functions above, next, the feature store tables need to be written out, first as Delta tables. These are the fundamental ‘offline’ data stores underpinning the feature store tables. We use the `FeatureStoreClient` client to create the feature tables, defining metadata like which database and table the feature store table will write to, and importantly, its key(s).

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
import pyspark.sql.functions as F
fs = FeatureStoreClient()

# COMMAND ----------

demographic_features_table = fs.create_feature_table(
  name = f"{database_name}.demographic_features",
  keys = "customerID",
  schema = demographics_df.schema,
  description = "Telco customer demographics")

# COMMAND ----------

service_features_table = fs.create_feature_table(
  name = f"{database_name}.service_features",
  keys = "customerID",
  schema = service_df.schema,
  description = "Telco service features")

# COMMAND ----------

compute_demographic_features.compute_and_write(input_data, feature_table_name = f"{database_name}.demographic_features", mode = "merge")

# COMMAND ----------

compute_service_features.compute_and_write(input_data, feature_table_name = f"{database_name}.service_features", mode = "merge")

# COMMAND ----------

inference_df.write.format("delta").saveAsTable(f"{database_name}.inference_data")

# COMMAND ----------


