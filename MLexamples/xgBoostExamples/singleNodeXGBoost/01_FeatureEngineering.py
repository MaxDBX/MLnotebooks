# Databricks notebook source
# MAGIC %md
# MAGIC #### Feature engineering
# MAGIC In the below command we do the following:
# MAGIC * For the categorical columns (inc. the label/output column) we create a stringIndexer, which converts the string values of the categorical columns to numerical values. We create a separate one for the label column
# MAGIC * We add the two stringIndexers together in a pyspark.ml.Pipeline object
# MAGIC * We "fit" the pipeline object to the dataset
# MAGIC * We subsequently transform the data-set using our pipeline object, and save it to a Feature Store
# MAGIC * **Note:** The Feature Store requires a primary key, so we artificially create an "id" column at the start.

# COMMAND ----------

# Artificially add a primary key
import pyspark.sql.functions as F
from pyspark.sql import Window

input_data = table('bank_db.bank_marketing')#.withColumn("primary_key", F.row_number().over(Window.orderBy()))

# COMMAND ----------

display(input_data)

# COMMAND ----------

def buildFeatureData(input_data):
  from pyspark.ml import Pipeline
  from pyspark.ml.feature import StringIndexer
  
  # define names/prefixes for the output columns
  label_col = "label"
  catInputCols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
  catOutputCols = [col + "_idx" for col in catInputCols]
  
  
  # pipeline stages (Converting string columns to indexed columns)
  string_indexer =  StringIndexer(inputCols = catInputCols, outputCols = catOutputCols)
  label_indexer = StringIndexer(inputCol = "y", outputCol = "label")
  stages = [string_indexer, label_indexer]
  pipeline = Pipeline(stages=stages)
  
  # Note: ONLY feature transformations that apply to BOTH train and test set should go here
  # Transformations that apply only to train set, and should be "fitted" to test set should be part of the model training itself
  # (For example: imputing null values of a column with its mean)

  pipelineModel = pipeline.fit(input_data)
  output_data = pipelineModel.transform(input_data)
  numericCols = ["age", "balance", "duration", "campaign", "previous", "day"]
    
  return output_data.select(numericCols + catOutputCols + [label_col])

# COMMAND ----------

feature_data = buildFeatureData(input_data)

# COMMAND ----------

display(feature_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store
# MAGIC Now that we have created our feature data set, we will store it into a feature store. Essentially the feature store is a delta-backed table with a number of extra features that work nicely with our features.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note: Edge cases
# MAGIC 
# MAGIC * How would the feature store take care of imputation?
