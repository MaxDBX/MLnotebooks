# Databricks notebook source
# MAGIC %md
# MAGIC #### Feature engineering
# MAGIC In the below command we do the following:
# MAGIC * For each of the categorical columns (inc. the label/output column) we create a stringIndexer, which converts the string values of the categorical columns to numerical values.
# MAGIC * We add the stringIndexers together in a pyspark.ml.Pipeline object
# MAGIC * We "fit" the pipeline object to the training data-set
# MAGIC * We subsequently transform the data-set using our pipeline object, and save it to Delta.

# COMMAND ----------

input_data = table('bank_db.bank_marketing')
cols = input_data.columns

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler

categoricalColumns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
  stages += [stringIndexer]
  
labelIndexer = StringIndexer(inputCol="y", outputCol="label")
stages += [labelIndexer]

# COMMAND ----------

display(input_data)

# COMMAND ----------

numericCols = ["age", "balance", "duration", "campaign", "previous", "day"]
feature_cols = [x + 'Index' for x in categoricalColumns] + numericCols

# COMMAND ----------

pipeline = Pipeline(stages=stages)
# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(input_data)
dataset = pipelineModel.transform(input_data)

# Keep relevant columns
selectedcols = ["label"] + feature_cols
dataset = dataset.select(selectedcols)
display(dataset)

# COMMAND ----------

dataset.write.mode('Overwrite').format("delta").saveAsTable("bank_db.bank_marketing_train_set")

# COMMAND ----------

