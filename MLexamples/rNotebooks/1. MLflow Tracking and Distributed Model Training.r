# Databricks notebook source
# MAGIC %md
# MAGIC ## MLflow Tracking and Distributed Model Training
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>
# MAGIC _____
# MAGIC **Note**: Make sure to run 00_PrepareData to make the dataset that is used in this notebook available
# MAGIC 
# MAGIC 
# MAGIC **Contents**
# MAGIC 
# MAGIC 1. Getting started with MLflow Tracking
# MAGIC 2. MLflow Model Registry
# MAGIC 3. Distributed Model Training with Spark and MLflow 
# MAGIC 
# MAGIC ##### Cluster Setup
# MAGIC * Make sure to use a cluster with the latest Databricks ML Runtime
# MAGIC * Install the following packages:
# MAGIC   * R packages: `ranger`, `carrier`, `mlflow`
# MAGIC   * Python packages: `Alembic`, `SQLAlchemy`
# MAGIC   
# MAGIC ___
# MAGIC 
# MAGIC ### Getting Started with MLflow Tracking
# MAGIC 
# MAGIC MLflow uses a tracking server to store data and artifacts associated with ML engineering work.  Data in the tracking server is organized into *experiments*, with each experiment containing separate *runs* for each iteration of model development.  Let's load the required R packages and set the experiment.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# DBTITLE 1,Load Packages
# Load packages
library(sparklyr)
library(dplyr)
library(ranger)
library(caret)
library(carrier)
library(mlflow)

# COMMAND ----------

# DBTITLE 1,Retrieve username and userhome
# MAGIC %run ./includes/mlSetup

# COMMAND ----------

# MAGIC %md
# MAGIC #### MLFlow set up for R on Databricks
# MAGIC To enable MLFlow for R on Databricks we are doing the following:
# MAGIC * First, we need to set the environment variables `MLFLOW_PYTHON_BIN` and `MLFLOW_BIN` to point towards the python and MLFlow python library and pexecutables, respectively. WE do this using the `Sys.setenv()` command. 
# MAGIC   * The reason we need to do this is that the R MLFlow library is essentially nothing more than a wrapper around the python MLFlow library. As such we want to make use of the pre installed Python MLFlow library.
# MAGIC * Secondly we use `mlflow_set_tracking_uri` to set the tracking URI, that is, the URI of our MLFlow tracking server, to "databricks", which means we point it to the tracking server that is part of this Databricks workspace.
# MAGIC 
# MAGIC #### Creating an experiment
# MAGIC After the initial set up, we are ready to use MLFlow! We can now make use of an MLFlow experiment. There are two ways to achieve this:
# MAGIC * Make use of the "default" experiment that is part of this notebook. To use this option we simply do not explicitly create a MLFlow experiment. MLFlow runs will be automatically tracked to the default experiment. This experiment can be accessed by clicking on "Experiment" in the top right of the notebook.
# MAGIC * Make use of an explicitly created experiment. This is what we do in this notebook. We denote a workspace path and artifact path (a DBFS location in which models and other objects associated with the experiment are saved), and pass it to the `set_or_create_mlflow_experiment` function. The experiment will be available in the workspace location.

# COMMAND ----------

# To avoid having to run mlflow::install_mlflow(), we set env vars
# to the existing Python installation and set the tracking URI to 'databricks'
Sys.setenv(MLFLOW_BIN = '/databricks/python/bin/mlflow')
Sys.setenv(MLFLOW_PYTHON_BIN = '/databricks/python/bin/python')
mlflow_set_tracking_uri("databricks")

# Create/set the experiment and optionally specify separate bucket for artifact storage
experiment_path <- sprintf("/Users/%s/R_singleNode", username)
artifact_path <- sprintf("%s/R_singleNode",userhome)

experiment_id <- set_or_create_mlflow_experiment(experiment_path,artifact_path)

## Set spark context for sparklyr, and change default table
sc <- spark_connect(method = "databricks")
tbl_change_db(sc,"rWorkshopDB")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading the data and performing feature engineering
# MAGIC In the below two cells we:
# MAGIC * Read the data using Spark, and convert the Spark data-frames to "normal" non-distributed R dataframes. We do this since we are making use of non-distributed R ML libraries, that take in normal R dataframes as input. We also split the data into a training and test data set.
# MAGIC * Feature engineering is kept simple in this notebook, since the focus is on machine learning. We do the following feature engineering steps:
# MAGIC   * Convert all categorical variables (including the label) to factors.
# MAGIC   * re-sample the training data to make the classes more balanced.

# COMMAND ----------

# DBTITLE 1,Read data and convert to non-distributed R dataframes
# Read the data from bank_marketing, and split them into train and test partitions of equal sizes (feel free to change the ratio!)
partitions <- spark_read_table(sc, "bank_marketing") %>% sdf_random_split(training = 0.5, test = 0.5)

# Convert to (single node) R dataframes
train_data <- as.data.frame(partitions$training)
test_data <- as.data.frame(partitions$test)

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# 0. Create label column
create_label <- function(df) {
  df <- df %>% mutate(label=ifelse(y == "yes",1,0)) %>% select(-y)
  df
}

# 1. Transform string variables to factor
to_factor <- function(df) {
  df <- df %>% mutate_if(is.character,factor)
  df <- df %>% mutate(label=as.factor(label))
  df}

# 2. Resample data
resample <- function(df, resample_factor) {
  df_0 <- df %>% filter(label == 0)
  df_1 <- df %>% filter(label == 1)
  
  count_0 <- nrow(df_0)
  count_1 <- nrow(df_1) 
  
  sample_0 <- df_0 %>% sample_n(resample_factor * count_1)
  bind_rows(sample_0, df_1)
}

# Make sure to only resample the train data.
final_train <- train_data %>% create_label() %>% resample(resample_factor = 2) %>% to_factor()
final_test <- test_data %>% create_label() %>% to_factor() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training with MLflow Tracking APIs
# MAGIC Let's train a random forest model on this dataset.  To be more efficient we'll define a training function to accept different hyperparameters and include MLflow APIs to log data and artifacts to the tracking server. Lets analyse the contents of this function a bit:
# MAGIC * The `with(mlflow_start_run(experiment_id), {}` command starts a run that is associated with the `experiment_id`. It works very similar to a file context manager. Within the `{}` block, we can interact with the MLFlow run. After the block, the run will be automatically closed. 
# MAGIC * Within the `{}` block we:
# MAGIC   * train a random forest model using the `ranger` package. 
# MAGIC   * Subsequently, we wrap it in a `crate` function, which provides a standardised way to get predictions with R models. 
# MAGIC   * Next, we use the `caret` package to get precision, recall and f1 scores of our model.
# MAGIC   * We can now log everything to our MLFlow run. We use `mlflow_log_param` to log the hyper parameters used, that is `num_trees`, `sample_fraction` and `node_size`. Subsequently we use `mlflow_log_metric` to log the accuracy metrics associated with our model, in our case the `precision`,`recall` and `f1` scores. Finally we use `mlflow_log_model` to save the crate-packaged random forest model in the dbfs location that's associated with the mlflow experiment (defined with `artifact_path` in cmd 6).

# COMMAND ----------

# Train model
train_model <- function(trainDF, testDF, num_trees, sample_fraction, node_size, mlflow_run_name) { 
  
  # With each training iteration
  with(mlflow_start_run(experiment_id = experiment_id), {
    
    # Train the model
    rf_model <- ranger(label ~ ., data = final_train,
                       num.trees = num_trees,
                       sample.fraction = sample_fraction,
                       min.node.size = node_size)
    
    # Package it up as an R function
    predictor <- crate(~stats::predict(rf_model, .x)$predictions, rf_model = rf_model)
    testDF$predictions <- predictor(testDF)
    
    # Extract our accuracy metrics
    precision <- caret::posPredValue(testDF$predictions,testDF$label, positive = "1")
    recall <- caret::sensitivity(testDF$predictions, testDF$label, positive = "1")
    f1_score = (2 * precision * recall) / (precision + recall)
    
    # Log the model parameters 
    mlflow_log_param("num_trees",num_trees)
    mlflow_log_param("sample_fraction",sample_fraction)
    mlflow_log_param("node_size",node_size)
    
    # Log the metrics 
    mlflow_log_metric("precision",precision)
    mlflow_log_metric("recall",recall)
    mlflow_log_metric("f1",f1_score)
    
    # Tag the run
    mlflow_set_tag("name", mlflow_run_name)
    
    # Log the model
    mlflow_log_model(predictor, "model")
    run_info <- mlflow_get_run()
  })
  return(run_info)
}

# COMMAND ----------

# MAGIC %md
# MAGIC We pass the prepared train and test sets to our `train_model` function. `run_info` is returned by `mlflow_get_run()` and gives us a data frame containing useful attributes related to the mlflow run. Have a look yourself!

# COMMAND ----------

run_info <- train_model(trainDF = final_train,
                        testDF = final_test,
                        num_trees = 300,
                        sample_fraction = 0.5,
                        node_size = 40,
                        mlflow_run_name = "Single Model Run")

# Leaving out nested columns with metrics, params, etc.
display(run_info[, 1:9])

# COMMAND ----------

# MAGIC %md
# MAGIC ___
# MAGIC ### Distributed Model Training with Spark and MLflow
# MAGIC 
# MAGIC In the previous section we trained one R model, and logged the hyper parameters, accuracy metrics and the model object itself to a MLFlow experiment. However, we are in a distributed environment, and as such we want to make use of the full power of Databricks clusters. Let's now explore how we can combine MLflow with Spark and `spark_apply` to train and track dozens of models in parallel!  
# MAGIC 
# MAGIC ##### Parallelised grid search with `spark_apply`
# MAGIC First, create a Spark DataFrame consisting of hyperparameters for the model.  Using `spark_apply` we can then create a model for each row in the hyperparameters DataFrame.  In our UDF we will authenticate to the MLflow tracking server and repeat the single node training code for each set of parameters.  
# MAGIC 
# MAGIC **Note:**
# MAGIC Make sure you create a unique ID column for your hyperparameter DataFrame.  We will use this as a grouping variable in `spark_apply`, ensuring that only one hyperparameter row is passed to the model training function.

# COMMAND ----------

# Create grid of hyperparameters
# We use expand.grid() to get every possible combinations of values
rf_grid <- expand.grid(num_trees = c(400,450,500,550,600),
                      min_node_size = c(10,15,20,25,30,35,45,50,55,60),
                      sample_fraction = c(0.4,0.45,0.5,0.55,0.6)) 

# Create unique ID 
rf_grid$id <- as.numeric(seq.int(nrow(rf_grid)))

# Push to Spark
sdf_rf_grid <- sdf_copy_to(sc, rf_grid, overwrite = TRUE)

display(rf_grid)

# COMMAND ----------

# MAGIC %md
# MAGIC Now define a user defined function (UDF) - `train_model_distributed` to repeat model training with MLflow tracking APIs in each task. As you'll note the UDF is very similar to the `train_model` function used to train a single model. There are two important differences: 
# MAGIC * We need to "tell" the worker nodes how to connect to the MLFlow tracking server. We do this by setting the `DATABRICKS_TOKEN` and `DATABRICKS_HOST` environment variables.
# MAGIC   * `DATABRICKS_TOKEN` is used to authenticate to the MLFLOW tracking server. We can use the "default" token that's associated with the notebook, which we can retrieve using `spark.databricks.token`.
# MAGIC   * `DATABRICKS_HOST` is to point to the URL of the mlflow tracking server, which in this case is simply the URL of the Databricks workspace. We can retrieve it using `spark.databricks.api.url`
# MAGIC * Supplying additional parameters to the UDF must be done using the "context" parameter. This context is simply a list containing all the additional parameters that we want to use. In this case it will contain the hyper parameters, train and test data, and the mlflow experiment ID

# COMMAND ----------

train_model_distributed <- function(sdf_grid_row, context){
  
  # Authenticate to hosted tracking server
  Sys.setenv(DATABRICKS_TOKEN = context$api_token)
  Sys.setenv(DATABRICKS_HOST = context$api_url)
  mlflow::mlflow_set_tracking_uri("databricks")
  
  # Point to existing Python installation of MLflow
  Sys.setenv(MLFLOW_BIN = '/databricks/python/bin/mlflow')
  Sys.setenv(MLFLOW_PYTHON_BIN = '/databricks/python/bin/python')
  
  # Train model: Note we get the train/test data from the context (which is passed as a list)
  rf_model <- ranger::ranger(label ~ ., data = context$train_data,
                          num.trees = sdf_grid_row$num_trees,
                          sample.fraction = sdf_grid_row$sample_fraction,
                          min.node.size = sdf_grid_row$min_node_size)
  
  # Package it up as an R function
  predictor <- carrier::crate(~stats::predict(rf_model, .x)$predictions, rf_model = rf_model)
  context$test_data$predictions <- predictor(context$test_data)
  
  # Calculate metrics and return row with metrics
  metrics_row <- dplyr::mutate(
    sdf_grid_row,
    precision = caret::posPredValue(context$test_data$predictions, context$test_data$label, positive = "1"),
    recall = caret::sensitivity(context$test_data$predictions, context$test_data$label, positive = "1"),
    f1_score = (2 * precision * recall) / (precision + recall))
  
  # Send params and metrics to the tracking server
  with(mlflow::mlflow_start_run(experiment_id = context$experiment_id),{
    mlflow::mlflow_log_param("single_model_num_trees", sdf_grid_row$num_trees)
    mlflow::mlflow_log_param("single_model_sample_fraction", sdf_grid_row$sample_fraction)
    mlflow::mlflow_log_param("node_size", sdf_grid_row$min_node_size)
    mlflow::mlflow_log_metric("single_model_precision", metrics_row$precision)
    mlflow::mlflow_log_metric("single_model_recall", metrics_row$recall)
    mlflow::mlflow_log_metric("single_model_f1", metrics_row$f1_score)  
    mlflow::mlflow_log_model(predictor, 'model')
    
    # Tag each run with a name using the 
    mlflow::mlflow_set_tag("name", paste("rf",
                                         sdf_grid_row$num_trees,
                                         sdf_grid_row$sample_fraction,
                                         sdf_grid_row$min_node_size, sep = "_"))
    run_info <- mlflow::mlflow_get_run()
    metrics_row$run_id <- as.character(run_info[[1,1]])})
  
  return(dplyr::select(metrics_row, 
                       run_id,
                       num_trees, min_node_size,
                       sample_fraction, precision, recall,
                       f1_score))
}

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will apply this function to each unique group of paramaters in our Spark DataFrame.  We also broadcast the training and test data as context variables, and specify the schema of our output DataFrame.  

# COMMAND ----------

# MAGIC %python
# MAGIC spark.conf.set("spark.sql.shuffle.partitions",80)

# COMMAND ----------

# Put together the pieces of our UDF
hyperopt_results <- spark_apply(
  x = sdf_rf_grid,
  f = train_model_distributed,
  group_by = 'id',
  context = list(train_data = final_train, test_data = final_test, 
                 api_token = spark.databricks.token, api_url = spark.databricks.api.url, experiment_id = experiment_id),
  columns = list(
    id = "numeric",
    run_id = "character",
    num_trees="numeric",
    min_node_size = "numeric",
    sample_fraction="numeric",
    precision = 'numeric',
    recall = 'numeric',
    f1_score = 'numeric')
  )

sdf_register(hyperopt_results, "hyperopt_tbl")
tbl_cache(sc, "hyperopt_tbl")
# Use collect to force Spark to evaluate the entire DataFrame and generate all the models
hyperopt_results %>% collect

# COMMAND ----------

# MAGIC %md
# MAGIC Subset the results to find the best model! 

# COMMAND ----------

# We can now retrieve the best parameters from the hyper_opt_df and retrain the model, and log it to MLFlow
best_model <- hyperopt_results %>% mutate(f1_rank = row_number(-f1_score)) %>% filter(f1_rank == 1) %>% collect
display(best_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ___
# MAGIC ### MLflow Model Registry
# MAGIC 
# MAGIC After experimenting and converging on a best model, we can push this model to the MLflow Model Registry.  This provides a host of benefits: <br><br>
# MAGIC 
# MAGIC * **One collaborative hub for model discovery and knowledge sharing**
# MAGIC * **Model lifecycle management tools to improve reliability and robustness of the deployment process**
# MAGIC * **Visibility and governance features for different deployment stages of each model**
# MAGIC 
# MAGIC Let's explore how to interact with the registry programmatically.  Please note that as of May 2020 we will need to use the Python API for this section. When working with more than one language we can make use of notebook workflows. That is: The python script for Model Registry 

# COMMAND ----------

# MAGIC %python
# MAGIC import mlflow
# MAGIC from mlflow.tracking.client import MlflowClient
# MAGIC 
# MAGIC run_id = "45f598d3782b4b19863c1fc4f9f162f3" # Copy RUN id from output of command 23!
# MAGIC 
# MAGIC client = MlflowClient()
# MAGIC # Register the model with the run URI and unique name
# MAGIC model_uri = "runs:/{}/model".format(run_id)
# MAGIC model_registry_id = spark.conf.get("com.databricks.tmp.uniqueid") # ensures we have a uniqueID for each student
# MAGIC model_registry_name = "bank_singlenode_{}".format(model_registry_id) 
# MAGIC 
# MAGIC model_details = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
# MAGIC 
# MAGIC # If we wanted to add another version to the registered model
# MAGIC version_id = client.search_model_versions("run_id='%s'" % run_id)[0].version

# COMMAND ----------

# DBTITLE 1,Transition old models in production to Archived stage
# MAGIC %python
# MAGIC model_list = client.search_model_versions("name='%s'" % model_registry_name)
# MAGIC 
# MAGIC version_prod_list = [x.version for x in model_list if x.current_stage == "Production"]
# MAGIC 
# MAGIC for version in version_prod_list:
# MAGIC   client.transition_model_version_stage(
# MAGIC     name=model_registry_name,
# MAGIC     version=version,
# MAGIC     stage="Archived")

# COMMAND ----------

# DBTITLE 1,Transition new model to Production stage
# MAGIC %python
# MAGIC client.transition_model_version_stage(
# MAGIC     name=model_registry_name,
# MAGIC     version=version_id,
# MAGIC     stage="Production"
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC Now load the registered model into memory using `mlfow_load_model` and the human-readable URI:

# COMMAND ----------

# Alternative URI schemes include runs:/ and the file system path
model_registry_name <- "bank_singlenode_3558a810-c8fc-4d8a-876e-9e3701d62413"
prod_model <- mlflow_load_model(model_uri = sprintf("models:/%s/production", model_registry_name))

## Generate prediction on 5 rows of data 
predictions <- data.frame(mlflow_predict(prod_model, data = test_data))
                          
names(predictions) <- "bank_term_pred"

## Take a look
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ___
# MAGIC At this point you should have a solid understanding of how to track your experimentation with MLflow, access the Model Registry, and consider the art of the possible with distributed training with Spark. 