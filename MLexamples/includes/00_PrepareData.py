# Databricks notebook source
# MAGIC %md
# MAGIC 1. Download the relevant data-set from the internet as CSV
# MAGIC 2. Convert the csv file to Delta format.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.Get dataset from the web
# MAGIC We use the wget command to download our data-set to the driver. After unzipping the file, we create a folder on dbfs, and copy our unzipped csv file there. Note that dbfs:/ is available as /dbfs/ on the driver (because it is a FUSE mount).

# COMMAND ----------

# MAGIC %sh wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip -O /tmp/bank.zip --no-check-certificate

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip the file to /tmp/bank

# COMMAND ----------

# MAGIC %sh unzip -o /tmp/bank.zip -d /tmp/bank

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new directory on dbfs:/, called BankMarketing

# COMMAND ----------

# MAGIC %fs mkdirs /BankMarketing

# COMMAND ----------

# MAGIC %md
# MAGIC Due to the FUSE mount, dbfs:/BankMarketing is now also available as /dbfs/BankMarketing on the driver. We can move our files there to make them available on dbfs.

# COMMAND ----------

# MAGIC %sh cp -rv /tmp/bank /dbfs/BankMarketing

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.Read data and save as Delta Table
# MAGIC Now that we have copied our csv to dbfs:/, the next step is to convert it to Delta format.

# COMMAND ----------

# MAGIC %md
# MAGIC Use spark.read to create a Spark Dataframe in which we will read our CSV data

# COMMAND ----------

bdf = spark.read.format("csv")\
.option("path", "dbfs:/BankMarketing/bank/bank-full.csv")\
.option("inferSchema", "true")\
.option("header", "true")\
.option("delimiter", ";")\
.option("quote", '"')\
.load()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS bank_db
# MAGIC LOCATION 'dbfs:/tmp/bank_data_db'

# COMMAND ----------

# MAGIC %md
# MAGIC We now write our spark dataframe to a delta table in max_db database, and call it bank_marketing. Note that, under the hood, the Delta table is saved somewhere on DBFS, i.e. on Azure Storage.

# COMMAND ----------

bdf.write.mode('Overwrite').format("delta").saveAsTable("bank_db.bank_marketing")

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/bank_data_db/bank_marketing"))