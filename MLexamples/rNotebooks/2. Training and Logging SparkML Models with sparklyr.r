# Databricks notebook source
# MAGIC %md
# MAGIC ## Training and logging SparkML models with sparklyr
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>
# MAGIC _____
# MAGIC **Note**: Make sure to run 00_PrepareData to make the dataset that is used in this notebook available
# MAGIC 
# MAGIC 
# MAGIC **Contents**
# MAGIC 
# MAGIC 1. Feature engineering and model training using sparklyr ML pipelines
# MAGIC 2. Logging sparklyr ML models with MLFlow
# MAGIC 3. Cross Validation with Sparklyr ML models
# MAGIC 
# MAGIC ##### Cluster Setup
# MAGIC * Make sure to use a cluster with the latest Databricks ML Runtime
# MAGIC * Install the following packages:
# MAGIC   * R packages: `ranger`, `carrier`, `mlflow`
# MAGIC   * Python packages: `Alembic`, `SQLAlchemy`
# MAGIC   
# MAGIC ___ 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up
# MAGIC * Load Packages
# MAGIC * Retrieve username and userhome and helper functions
# MAGIC * Get spark connection
# MAGIC * Set up MLFlow experiment
# MAGIC * Retrieve data set.

# COMMAND ----------

# DBTITLE 1,Load Packages
# Load other packages
library(purrr)
library(sparklyr)
library(dplyr)
library(mlflow)
library(ggplot2)

# COMMAND ----------

# DBTITLE 1,Retrieve username and userhome and utility functions
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

Sys.setenv(MLFLOW_BIN = '/databricks/python/bin/mlflow')
Sys.setenv(MLFLOW_PYTHON_BIN = '/databricks/python/bin/python')
mlflow_set_tracking_uri("databricks")

# Create/set the experiment and optionally specify separate bucket for artifact storage.
experiment_path <- sprintf("/Users/%s/R_sparklyrML", username)
artifact_path <- sprintf("%s/R_sparklyrML",userhome)

experiment_id <- set_or_create_mlflow_experiment(experiment_path, artifact_path)

# COMMAND ----------

# MAGIC %python
# MAGIC spark.conf.set("spark.sql.shuffle.partitions", sc.defaultParallelism)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get data

# COMMAND ----------

# MAGIC %run ./includes/prepareBankData

# COMMAND ----------

# Set spark context for sparklyr, and change default table
sc <- spark_connect(method = "databricks")
tbl_change_db(sc, "rWorkshopDB")

partitions <- spark_read_table(sc, "bank_marketing") %>%
sdf_random_split(training = 0.5, test = 0.5)

train_data <- partitions$training
test_data <- partitions$test

# COMMAND ----------

# MAGIC %md 
# MAGIC ####ML Pipelines with Sparklyr
# MAGIC 
# MAGIC With Sparklyr machine learning works in much the same way as sklearn from Python. One of the main features of ML in sparklyr is the `ml_pipeline` object, which you can use to "bundle" various feature engineering operations and machine learning models together in one object that can be interacted with. In the command below, have a look at the `build_rf_pipeline` function. To build the pipeline we do the following:
# MAGIC * Create a pipeline object by calling `ml_pipeline(sc)` , where sc is the sparklyr spark connection
# MAGIC * We then "add" feature transformation stages to our pipeline object between lines 32 and 45. Specifically:
# MAGIC   * We apply `ft_string_indexer` to each of the character columns, to convert them to numerical valued columns. The reason for this is that the SparklyrML models do not handle character vectors. By default, the values are ordered by label frequency.
# MAGIC   * Subsequently we use `ft_vector_assembler` to group all the feature columns into a feature vector, which is needed for the subsequent steps.
# MAGIC   * Then, we apply principal component analysis, which is a feature reduction technique. 
# MAGIC     * **Note** PCA is technically not needed since the random forest algorithm selects the most important features anyway as part of the training algorithm. It is used here as a way to show what is possible with sparklyrML pipelines
# MAGIC * As a last step we add the `ml_random_forest_classifier()` to the ML pipeline.
# MAGIC 
# MAGIC #### Fitting the pipeline

# COMMAND ----------

## FUNCTIONS

# Function for resampling (very basic)
resample <- function(sdf,multiplier){
  sdf_0 <- sdf %>% filter(y == "no")
  sdf_1 <- sdf %>% filter(y == "yes")
  
  count_0 <- sdf_nrow(sdf_0)
  count_1 <- sdf_nrow(sdf_1)
  
  sample_0 <- sdf_0 %>% sdf_sample(multiplier*count_1/count_0)
  sdf_bind_rows(sdf_1,sample_0)
}


# Function to build pipeline
build_rf_pipeline <- function(data_set, params, labelCol = "y") {
  
  # Initiate sparklyr ml_pipeline object
  model_pipeline <- ml_pipeline(sc) 
  
  # Retrieve all columns for a given type from the dataset
  get_type_cols <- function(data_set,type_filter){
    type_cols <- colnames(data_set %>% select_if(type_filter))
  }
  
  character_cols <- get_type_cols(data_set %>% select(-labelCol),is.character)
  numerical_cols <- get_type_cols(data_set %>% select(-labelCol),is.numeric)
  
  # Convert all categorical columns to numerical values using ft_string_indexer
  for(col in {character_cols}){
    model_pipeline <- ft_string_indexer(model_pipeline,
                                        input_col = col,
                                        output_col = paste0('in','_',col))
  }
  
  # Create feature vector from all columns, and apply principal component analysis.
  # Finally convert label column to numeric values as well.
  model_pipeline <- model_pipeline %>%
  ft_vector_assembler(c(paste0('in_',character_cols), numerical_cols), 'features') %>% 
  ft_pca(input_col = 'features',output_col = 'out_features', k = params$pca_num, uid = 'pca_1') %>%
  ft_string_indexer(input_col = labelCol, output_col = "label")
  
  # Add random forest classifier to the ML pipeline.
  model_pipeline <- model_pipeline %>% 
  ml_random_forest_classifier(label_col = 'label',
                              features_col = 'features', 
                              num_trees = params$num_trees,
                              subsampling_rate = params$subsampling_rate,
                              max_depth = params$max_depth, 
                              impurity = params$impurity,
                              uid = 'random_forest_1')
  
  return(model_pipeline)}

fit_pipeline <- function(model_pipeline, data_set) {
  model <- ml_fit(model_pipeline, data_set)
  crate_model <- crate(~sparklyr::ml_transform(model, .x), model = model)
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluating Spark ML pipelines
# MAGIC In the below command we created a number of functions to evaluate our fitted pipeline model, and subsequently to log it to MLFlow. Lets have a closer look.
# MAGIC * The `get_metric` function is a helper function that uses `sparklyr` ( = `dplyr`) methods to calculate f1 score, precision and recall given a data set with labels and predictions.
# MAGIC * The `log_pipeline` function does the following:
# MAGIC   * It starts up a mlflow run context using `with(mlflow_start_run() {} )`
# MAGIC   * It logs all the hyper parameters used to build the ML model to the MLFlow run.
# MAGIC   * We generate a predictions dataframe using our crate model, and do some processing on it to get it in the right format.
# MAGIC   * We generate the metrics using the aforementioned `get_metrics` function.
# MAGIC   * We get the auc number, and subsequently a AUC plot, which is logged as an artifact to the MLFlow run.

# COMMAND ----------

# Get precision/recall/f1
get_metrics <- function(pred,positive = 1) {
  c_matrix <- pred %>% group_by(label,prediction) %>% tally() %>% collect
  
  positive_total <- sum(c_matrix %>% filter(label == positive) %>% select(n))
  pred_positive_total <- sum(c_matrix %>% filter(prediction == positive) %>% select(n))
  label_and_pred_positive <- sum(c_matrix %>% filter(label == positive & prediction == positive) %>% select(n))
  
  metrics <- list()
  metrics$P <- label_and_pred_positive/pred_positive_total
  metrics$R <- label_and_pred_positive/positive_total
  metrics$F1 <- 2 * metrics$P * metrics$R /(metrics$P + metrics$R)
  
  return(metrics)
}


# Function to evaluate pipeline
log_pipeline <- function(crate_model,test_data,params,mlflow_run_name,experiment_id){
  
  with(mlflow_start_run(experiment_id = experiment_id), {
    
    # log hyper parameters
    for (param_name in names(params)) {
      mlflow_log_param(param_name,params[[param_name]])
    }
    
    # Generate predictions
    predictions <- crate_model(test_data)
    predictions <- predictions %>% 
    mutate(label = as.double(label),
           prediction = as.double(prediction)) %>%
    sdf_separate_column('probability', into = c("prob_0","prob_1"))
    
    # Get precision/recall/f1 metrics from our predictions
    metrics <- get_metrics(predictions)
    
    # get aoc score from our predictions:
    aoc_number <- ml_binary_classification_eval(predictions, "label","prob_1")
    
    # plot aoc curve, log it to experiment:
    aoc_data <- get_aoc_data(predictions)
    
    p <- ggplot(aoc_data %>% collect, aes(x = fpr, y = tpr)) + 
    geom_point(size=2, color = 'red') +
    theme_bw()
    
    # Need to save the plot to temp location before we can log it to mlflow
    tmp <- tempfile(fileext = ".png")
    ggsave(tmp,p, device = png())
    mlflow_log_artifact(tmp, "aoc.png")
    file.remove(tmp)
    
    # Log metrics to MLFlow
    mlflow_log_metric('f1',metrics$F1)
    mlflow_log_metric('Precision',metrics$P)
    mlflow_log_metric('Recall',metrics$R)
    mlflow_log_metric("AOC",aoc_number)
    
    # log the crate model to mlflow
    mlflow_log_model(crate_model,"model")
    
    mlflow_set_tag("name",mlflow_run_name)
    
    run_info <- mlflow_get_run()
  })
  return(run_info)
}

# COMMAND ----------

# Configure parameters
params <- list()
params$label_0_to_1_ratio <- 2
params$pca_num = 8
params$num_trees = 500
params$subsampling_rate = 0.5
params$max_depth = 6
params$impurity = 'gini'

# Resample train data set
sample_train <- resample(train_data,params$label_0_to_1_ratio)

# Build + fit pipeline
pipeline <- build_rf_pipeline(sample_train,params)

model <- fit_pipeline(pipeline,sample_train)

# COMMAND ----------

# Evaluate pipeline model
predictions <- log_pipeline(model, test_data,params, 'single_sparklyr_model',experiment_id = experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grid Search
# MAGIC * Note how we build the hyperparameter grid:
# MAGIC * We have defined ID's for the pipeline steps that we want to want to iterate(pca_1, and random_forest_1).
# MAGIC * For each of these pipeline steps we can then choose
# MAGIC * Note: Included principal component analysis, but this is actually note needed with RF. However it is interesting to see that you can tune all kinds of pipeline parametes in cross-validation. Any ideas for a different pipeline parameter I could use to showcase this?

# COMMAND ----------

fit_cv_model <- function(pipeline, train_data, hyper_param_grid, num_folds, parallelism) {
  
  my_cv <- ml_cross_validator(
    sc, estimator = pipeline, estimator_param_maps = grid,
    evaluator = ml_binary_classification_evaluator(sc),
    num_folds = num_folds,
    parallelism = parallelism)
  
  cv_model <- ml_fit(my_cv,train_data)
  
  cv_metrics <- ml_validation_metrics(cv_model)
  
  # package up the best model
  crate_best_model <- crate(~sparklyr::ml_transform(model, .x),model=cv_model$best_model)
  
  # get parameters of best model for logging later
  params <- list()
  params$label_0_to_1_ratio <- 2
  params$pca_num = cv_metrics$k_2[[1]]
  params$num_trees = cv_metrics$num_trees_1[[1]]
  params$subsampling_rate = cv_metrics$subsampling_rate_1[[1]]
  params$max_depth = 6 # didn't evaluate this one in our cross validation
  params$impurity = cv_metrics$impurity_1[[1]]
  
  results = list(crate_best_model = crate_best_model, params = params)
  return(results)
}

# COMMAND ----------

# Build grid of hyperparameters: We are evaluating principal components, subsampling rates, num_trees and impurity strategy
grid <- list(
  pca_1 = list(k = c (8,10,12)),
  random_forest_1 = list(
    subsampling_rate = c(0.4,0.6),
    num_trees = c(500,600,700),
    impurity = c("entropy", "gini")))

# build cross evaluator
my_cv <- ml_cross_validator(
  sc, estimator = pipeline, estimator_param_maps = grid,
  evaluator = ml_binary_classification_evaluator(sc),
  num_folds = 3,
  parallelism = 5)

# COMMAND ----------

results <- fit_cv_model(pipeline,sample_train, grid, num_folds = 3, parallelism = 5)

# COMMAND ----------

run_info <- evaluate_pipeline(results$crate_best_model, test_data, results$params,mlflow_run_name = "cv_model", experiment_id = experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### todo: add in mlflow_load_model example + run it on some data

# COMMAND ----------

