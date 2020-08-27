# Databricks notebook source
# MAGIC %md
# MAGIC ### Spark ML example
# MAGIC In this notebook we show a simple spark ML model, that we also use with mlflow and mlflow model registry.
# MAGIC * *Note:* This notebook builds off of the data-set that was created in notebook 00. Make sure to run that first! 
# MAGIC * The logged model can be used in modelRegistry exactly the same way as was done in the many models example.
# MAGIC * Deploying in a Databricks notebook also works exactly the same
# MAGIC * *If you want to deploy to Sagemaker:* You will need to log your model to *mleap* instead 
# MAGIC   *To do this use `import mlflow.mleap` and subsequently log the model using `mlflow.mleap.log_model`

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 0: Create or Set an MLFlow experiment
# MAGIC Make sure to set an experiment that corresponds with your own user name

# COMMAND ----------

import mlflow
user_home = "FILL_IN" # fill in your user home. This is most likely to be the email address used for logging in
mlflow.set_tracking_uri("databricks") # if databricks -> then 'MANAGED' and somewhere on the control plane
experimentPath = "/Users/{}/mlflowExperiments/bank_sparkML".format(user_home) # workspace path by default
mlflowModelSaveDir = "/Users/{}/mlflowExperiments/bank_sparkML".format(user_home) #DBFS path where models and other artifacts will be saved

# COMMAND ----------

def setOrCeateMLFlowExperiment(experimentPath,mlflow_model_save_dir):
  from mlflow.exceptions import MlflowException
  try:
    experiment_id = mlflow.create_experiment(experimentPath, "dbfs:" + mlflow_model_save_dir)
  except MlflowException: # if experiment is already created
    experiment_id = mlflow.get_experiment_by_name(experimentPath).experiment_id
    mlflow.set_experiment(experimentPath)
  return experiment_id

# COMMAND ----------

experiment_id = setOrCeateMLFlowExperiment(experimentPath,mlflowModelSaveDir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Read and inspect the data

# COMMAND ----------

input_data = table('max_db.bank_marketing')

# COMMAND ----------

display(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define functions for SparkML pipeline and Cross Validation
# MAGIC * To train a Spark Random Forest model, we are using the Spark ML pipeline framework. For more documentation see [here](https://spark.apache.org/docs/latest/ml-pipeline.html)
# MAGIC * Using MLFlow: We will give cross validation a 

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def buildPipeline(numTrees, maxDepth):
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
  
  ## Add RandomForestClassifier to the pipeline
  rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, numTrees=20, featureSubsetStrategy="all", seed=42, maxBins=100)
  stages += [rf]
  
  ## Create the pipeline
  pipeline = Pipeline(stages = stages)
  
  ## build a parameter grid for the Cross Validator:
  param_grid = (ParamGridBuilder()
                .addGrid(rf.numTrees, numTrees)
                .addGrid(rf.maxDepth, maxDepth)
                .build())

  cv = CrossValidator(estimator = pipeline, estimatorParamMaps=param_grid, numFolds = 3, seed = 42, 
                      parallelism = 4, evaluator =BinaryClassificationEvaluator())  
  return cv  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit + evaluate model and log to MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC Having defined all our functions, we are now ready to execute our spark pipeline. In here, we use MLFlow as follows
# MAGIC * We use `mlflow.log_param()` to log the number of trees and max Depth we iterate over in the cross validation
# MAGIC * We use `mlflow.log_metric()` to log the metric of the best model from our cross validation using
# MAGIC * We use `mlflow.spark.log_model()`to log (save) our best model to the MLFlow experiment. The save location will be in a DBFS (= ADFS) location which is associated with your MLFlow experiment (defined in command 3)

# COMMAND ----------

(trainingData, testData) = input_data.randomSplit([0.8, 0.2], seed = 42)

# COMMAND ----------

# Define our evaluator object
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

# COMMAND ----------

import mlflow.spark
numTrees = [100,200,300]
maxDepth = [4,5,6]

# Fit model with train data and evaluate it with test data
with mlflow.start_run(experiment_id = experiment_id):
  # Log the number of trees and maxDepth we iterate over
  mlflow.log_param("numTrees",numTrees)  
  mlflow.log_param("maxDepth",maxDepth)
  cvModel = buildPipeline(numTrees,maxDepth)
  rfModel = cvModel.fit(trainingData)
  test_metric = evaluator.evaluate(rfModel.transform(testData))
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric) # Log mlflow metric: F1 score
  mlflow.spark.log_model(spark_model=rfModel.bestModel,artifact_path="best_model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inspecting your model run in MLFlow
# MAGIC Now that we performand a cross validation run, you can do to your MLFlow experiment (in the workspace as defined with `experiment_path`) and check out your run! Also note down the run ID (click on the model, and inspect the string of characters to the right of Run), as we will use this to add the model to Model Registry.

# COMMAND ----------

