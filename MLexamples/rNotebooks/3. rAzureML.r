# Databricks notebook source
# MAGIC %md
# MAGIC #### Azure ML for R
# MAGIC In this notebook we will use MLFLow and the azureml sdk for R package to:
# MAGIC * Load a model from MLFlow model registry.
# MAGIC * Add this model to AzureML model registry.
# MAGIC * Attach a AKS cluster to AzureML
# MAGIC * Deploy the model to an AKS cluster as a AzureML web service
# MAGIC * Run predictions

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0. Libraries and set-up
# MAGIC We will need the following libraries:
# MAGIC * azure ml sdk for R. This should be installed directly from github using `remotes::install_github()`, as the version on CRAN seems to not be the latest version.
# MAGIC * Other libraries can be installed through the cluster UI:
# MAGIC   * `mlflow` (R) (for loading the MLFlow model)
# MAGIC   * `SQLAlchemy` (Python) (dependency for MLFlow)
# MAGIC   * `Alembic` (Python) (dependency for MLFlow)
# MAGIC   * `azureml-sdk[databricks]` (Python) (Dependency for R azuremlsdk package)
# MAGIC   * (if you want to run the model in this notebook: install ranger and carrier as well)
# MAGIC   
# MAGIC ##### MLFlow configuration:
# MAGIC To get MLFlow to work with R, we must point the environment variables `MLFLOW_PYTHON_BIN` and `MLFLOW_BIN` to the python executable and python MLFlow executable respectively. This is done in cmd 5.

# COMMAND ----------

remotes::install_github('https://github.com/Azure/azureml-sdk-for-r')

# COMMAND ----------

library(mlflow)
Sys.setenv(MLFLOW_BIN = '/databricks/python/bin/mlflow')
Sys.setenv(MLFLOW_PYTHON_BIN = '/databricks/python/bin/python')
mlflow_set_tracking_uri("databricks")

# COMMAND ----------

library(azuremlsdk)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Retrieve AzureML workspace
# MAGIC To deploy a model to AzureML, we need to be able to interact with the AzureML workspace. To do this we need to retrieve a AzureML workspace object through the `get_workspace()` command. For authentication, we use a service principal object that we get through `service_principal_authentication()`, in which we fill in the credentials of a service principal (that you should have set up in Azure).

# COMMAND ----------

# DBTITLE 1,Retrieve Service Principal object
svc <- service_principal_authentication(
  tenant_id=dbutils.secrets.get("fieldeng","mthone_tenant_id"),
  service_principal_id=dbutils.secrets.get("fieldeng","mthone_sp_id"),
  service_principal_password=dbutils.secrets.get("fieldeng","mthone_sp_pw")
)

# COMMAND ----------

# DBTITLE 1,Retrieve AzureML workspace object
# Authenticate to ML workspace
ws <- get_workspace(
  name = "mthoneML",
  auth = svc,
  subscription_id=dbutils.secrets.get("fieldeng","mthone_subscription_id"),
  resource_group = "mthone-fe"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Attach a Azure Kubernetes cluster to the Azure ML workspace
# MAGIC We use `attach_aks_compute` to attach an Azure Kubernetes cluster to the AzureML workspace. Note that this step can also be done in the Azure portal. If a AKS cluster is already attached to the workspace, you can retrieve its details using the `get_compute` command. 

# COMMAND ----------

aks_cluster_name <- "mthoneAKS" # Name of an existing AKS cluster

aks_compute_obj <- attach_aks_compute(ws, 
                                      resource_group = "mthone-fe",
                                      cluster_name = aks_cluster_name)

wait_for_provisioning_completion(aks_compute_obj, show_output = TRUE)

# COMMAND ----------

aks_compute_obj <- get_compute(ws, cluster_name = aks_cluster_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Create an environment
# MAGIC AzureML works with "environments", which are a way to specify what the docker image that runs on AzureML should look like. We use `r_environment()` to specify a R environment that should contain the `carrier` and `ranger` packages, which are needed to run the model.

# COMMAND ----------

r_env <- r_environment(name = "deploy_env_2", cran_packages = list(list(name ="carrier"),list(name="ranger"), list(name = "jsonlite")))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Create an entry script
# MAGIC Next, we need to create an "entry script". This is the script that will be used to handle the `GET` request of getting productions. By convention, it must contain an init function that returns a function which takes in the data in json format, and returns the predictions in json format. In the below command we generate this init script and save it to a local temporary directory.

# COMMAND ----------

# "entry script". This script will be run inside the AKS cluster to generate predictions.
entry_script_dir <- "dbfs:/tmp/mthone/"
entry_script_name <- "score.R"

dbutils.fs.put(paste0(entry_script_dir,entry_script_name), "
library(jsonlite)
library(carrier)
library(ranger)

init <- function()
{
  model_path <- Sys.getenv(\"AZUREML_MODEL_DIR\")
  model <- readRDS(file.path(model_path, \"cratemodel\"))
  message(\"model is loaded\")
  
  function(data)
  {
    input_data <- as.data.frame(fromJSON(data))
    prediction <- model(input_data)
    result <- as.character(prediction)
    toJSON(result)
  }
}", overwrite = TRUE)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Generate an inference configuration
# MAGIC We use the inference_config method to create an inference configuration for deploying the model. It contains the path to the entry script, as well as the environment that we want to use. 
# MAGIC **Note**: Under the hood, inference_config will generate a python version of the entry script. To ensure it ends up in the same directory as the R script (which must happen for the deployment to work correctly), make sure to provide the `source_directory` (the directory where our entry script is located) explicitly.

# COMMAND ----------

# Create the inference config. Make sure you specify source directory, else it will fail (it's because it generates a python scoring script, 
# and it will end up in a different location if you don't specify the source directory)
# NOTE: we use /dbfs/ instead of dbfs:/ since we make use of the FUSE mount.
i_conf <- inference_config(entry_script = entry_script_name,
                              source_directory = paste0("/",str_replace(entry_script_dir,":","")),
                              environment = r_env)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Load our model from MLFlow and register it to AML
# MAGIC We get our model from MLFlow model registry (but you can also of course just get it from a MLFlow experiment), and save it locally as a `.RDS` file. We then add this model to the AzureML model registry, as this is convenient when deploying the model to a AzureML webservice.  
# MAGIC **Note**:  
# MAGIC You of course do not need MLFlow model registry in this case. You can also retrieve it directly from the MLFlow experiment using `mlflow_load_model("runs:/RUN_ID/model")`, or perhaps it was already added to AML model registry in a different notebook (in which you can skip cmd 19/20 altogether)

# COMMAND ----------

model_path <- "/dbfs/tmp/carsten.thone@databricks.com/cratemodel"

model <- mlflow_load_model("models:/bank_singlenode_3558a810-c8fc-4d8a-876e-9e3701d62413/production")

saveRDS(model, model_path)

aml_model <- register_model(ws,
                        model_path = model_path,
                        model = "cratemodel",
                        description = "predict bank deposits")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7. Deploy the model to AzureML as a web service
# MAGIC We use the `deploy_model` command to deploy the model from the AzureML model registry to our Azure Kubernetes cluster. We can use `get_webservice_logs()` to retrieve details of the deployment. In case you already have a deployment, you can use `get_webservice` to retrieve it.

# COMMAND ----------

web_service_name <- "aks-service-1"

aks_config <- aks_webservice_deployment_config(cpu_cores = 1, memory_gb =1)

aks_service <- deploy_model(ws,
                            web_service_name,
                            models = list(aml_model),
                            inference_config = i_conf,
                            deployment_config=aks_config,
                            deployment_target = aks_compute_obj)

# COMMAND ----------

aks_service <- get_webservice(ws, web_service_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8. Run predictions! 
# MAGIC We are now ready to interact with our web service. Below we create an example dataframe, convert it to json, and we use `invoke_webservice` to generate predictions.

# COMMAND ----------

example_data <- data.frame(
  age = c(18),
  job = c("student"),
  marital = c("single"),
  education = c("primary"),
  default= c("no"),
  balance = c(608),
  housing = c("no"),
  loan = c("no"),
  contact = c("cellular"),
  day = c(12),
  month= c("aug"),
  duration = c(267),
  campaign = c(1),
  pdays = c(-1),
  previous= c(0),
  poutcome= c("unknown"))

# COMMAND ----------

library(jsonlite)
predicted_val <- invoke_webservice(aks_service, toJSON(example_data))

# COMMAND ----------

predicted_val