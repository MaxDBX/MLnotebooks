# Databricks notebook source
# MAGIC %run ../includes/00_PrepareData

# COMMAND ----------

input_data = table('bank_marketing')

# COMMAND ----------

import pyspark.sql.functions as f

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

def buildPipeline(dataSet):
  categoricalColumns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
  stages = [] # stages in our Pipeline
  for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
    stages += [stringIndexer]
  
  # Combine all features into a "features" column
  numericCols = ["age", "balance", "campaign", "previous", "day"]
  assemblerInputs = numericCols + [c + "Index" for c in categoricalColumns]
  assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
  stages += [assembler]
  
  # Index the label.
  labelIndexer = StringIndexer(inputCol="y", outputCol="label")
  stages += [labelIndexer]
  
  # build pipeline
  pipeline = Pipeline(stages = stages)
  pipelineModel = pipeline.fit(dataSet)
  
  return pipelineModel
  
  # fit + transform our data.

# COMMAND ----------

(trainingData, testData) = input_data.randomSplit([0.8, 0.2], seed = 42)

# COMMAND ----------

pipelineModel = buildPipeline(trainingData)

transformedTrainData = pipelineModel.transform(trainingData).withColumn("label",f.col("label").cast("int")).cache()
transformedTestData = pipelineModel.transform(testData).withColumn("label",f.col("label").cast("int")).cache()

# COMMAND ----------

transformedTrainData.createOrReplaceGlobalTempView("globalTempTrainData")
transformedTestData.createOrReplaceGlobalTempView("globalTempTestData")