# Databricks notebook source
# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

database_name = 'max_db'
bronze_table_name = "bronze_telco_churn"

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
driver_to_dbfs_path = 'dbfs:/home/{}/ibm-telco-churn/Telco-Customer-Churn.csv'.format(user)
dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# COMMAND ----------

# Delete the old database and tables if needed
spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")

# COMMAND ----------

# Create new DB
spark.sql(f"CREATE DATABASE {database_name}")

# COMMAND ----------

from pyspark.sql.types import *

schema = StructType([
  StructField('customerID', StringType()),
  StructField('gender', StringType()),
  StructField('seniorCitizen', DoubleType()),
  StructField('partner', StringType()),
  StructField('dependents', StringType()),
  StructField('tenure', DoubleType()),
  StructField('phoneService', StringType()),
  StructField('multipleLines', StringType()),
  StructField('internetService', StringType()), 
  StructField('onlineSecurity', StringType()),
  StructField('onlineBackup', StringType()),
  StructField('deviceProtection', StringType()),
  StructField('techSupport', StringType()),
  StructField('streamingTV', StringType()),
  StructField('streamingMovies', StringType()),
  StructField('contract', StringType()),
  StructField('paperlessBilling', StringType()),
  StructField('paymentMethod', StringType()),
  StructField('monthlyCharges', DoubleType()),
  StructField('totalCharges', DoubleType()),
  StructField('churnString', StringType())
  ])

# COMMAND ----------

# Read CSV, write to Delta and take a look
bronze_df = (spark.read.format('csv').schema(schema).option('header','true')
             .load(driver_to_dbfs_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Basic data cleaning
# MAGIC We assume the data was already cleaned. The below step is not considered part of the feature engineering step. Rather it's part of the "cleaning" process

# COMMAND ----------

bronze_df.write.format('delta').mode('overwrite').saveAsTable(database_name + "." + bronze_table_name)
