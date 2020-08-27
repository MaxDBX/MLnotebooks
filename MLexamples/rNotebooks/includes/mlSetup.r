# Databricks notebook source
# MAGIC %scala
# MAGIC val tags = com.databricks.logging.AttributionContext.current.tags
# MAGIC val name = tags.getOrElse(com.databricks.logging.BaseTagDefinitions.TAG_USER, java.util.UUID.randomUUID.toString.replace("-", ""))
# MAGIC val username = if (name != "unknown") name else dbutils.widgets.get("databricksUsername")
# MAGIC val userhome = s"dbfs:/tmp/$username"
# MAGIC 
# MAGIC spark.conf.set("com.databricks.tmp.username", username)
# MAGIC spark.conf.set("com.databricks.tmp.userhome", userhome)
# MAGIC 
# MAGIC display(Seq())

# COMMAND ----------

# MAGIC %python
# MAGIC import uuid
# MAGIC uniqueId = uuid.uuid4()
# MAGIC spark.conf.set("com.databricks.tmp.uniqueid", str(uniqueId))

# COMMAND ----------

userhome <- SparkR::sparkR.conf()$com.databricks.tmp.userhome
username <- SparkR::sparkR.conf()$com.databricks.tmp.username   
uniqueid <- SparkR::sparkR.conf()$com.databricks.tmp.uniqueid   # use this for model registry

# COMMAND ----------

# DBTITLE 1,Utility Functions
# Set experiment if it exists, otherwise create a new one.
set_or_create_mlflow_experiment <- function(experiment_path, artifact_path) {
  experiment_id <- tryCatch(
    {
      mlflow_create_experiment(name = experiment_path, artifact_location = artifact_path)
    },
    error = function(e)
    {
      mlflow_set_experiment(experiment_path)
    })
  return(experiment_id)
}

# get AOC curve/value
get_aoc_data <- function(predictions, flipped_prob = TRUE, rounding = 2) {
  # unpack predictions
  aoc_data <- predictions %>% sdf_separate_column('probability')
  # unpacked probability comes back as probability_1
  
  # For some reason the rf classifier returns inverted probabilities. 
  # These need to be flipped for correct analysis
  if(flipped_prob == TRUE)
  {
    aoc_data <- aoc_data %>% mutate(probability_1 = 1-probability_1)
  }
  
  aoc_data <- aoc_data %>% 
  mutate(
    probability_1 = round(probability_1,rounding)) %>% 
  group_by(probability_1) %>% 
  summarise(counter = n(),
            pos_label = sum(label)) %>%
  arrange(1-probability_1) %>%
  mutate(
    fpr = cumsum(counter-pos_label)/(sum(counter)- sum(pos_label)),
    tpr = cumsum(pos_label)/(sum(pos_label))) %>% 
  select(probability_1,fpr,tpr)
  return(aoc_data)}

# Get AOC number (basically doing a right-hand summation)
get_aoc_number <- function(aoc_data) {
  aoc_data <- aoc_data %>% arrange(fpr) %>%mutate(
    fpr_diff = lead(fpr,1) - fpr,
    aoc_sq = tpr * fpr_diff
  )
  return((aoc_data %>% summarise(sum(aoc_sq)) %>% collect)[[1]])}