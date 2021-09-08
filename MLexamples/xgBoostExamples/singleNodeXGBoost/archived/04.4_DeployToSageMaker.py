# Databricks notebook source
# MAGIC %md
# MAGIC ### Deploying our model to AWS Sagemaker
# MAGIC In this notebook we will deploy our model from "production" stage in model registry to AWS Sagemaker. In order for this notebook to work, a few prerequisites must be met:
# MAGIC * First we need to build a docker image that will work with our MLFlow model. To do this:
# MAGIC   * Install docker and MLFlow on your local machine
# MAGIC   * use `mlflow sagemaker build-and-push --build` to create an MLFlow image that will work with Sagemaker
# MAGIC * Secondly, the docker image must be uploaded to AWS Elastic Container Registry (ECR)
# MAGIC   * In your AWS account, create a new ECR repository
# MAGIC   * [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html), and subsequently [configure credentials](https://docs.aws.amazon.com/cli/latest/reference/configure/)
# MAGIC   * Subsequently, [you must authenticate docker with your ECR repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Registries.html#registry_auth)
# MAGIC   * You must also make sure the AWS [user used for AWS CLI has access to ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/security_iam_id-based-policy-examples.html).
# MAGIC   * Then, use `docker tag` to rename your image to the ECR repository link.
# MAGIC   * Finally, use `docker push` to push the image to your ECR repository
# MAGIC * Third, we need to make sure our Databricks cluster is authenticated with AWS Sagemaker:
# MAGIC   * To do this, get or create an IAM role that has sagemaker permissions. AmazonSageMakerFullAccess role in AWS IAM is readily available
# MAGIC   * Secondly, add this role to the cross-account role that has been defined when setting up Databricks.
# MAGIC     * [this link](https://docs.databricks.com/administration-guide/cloud-configurations/aws/instance-profiles.html) roughly shows the steps. Instead of adding an S3 IAM role, you instead add the AmazonSageMakerFullAccess role.
# MAGIC   * Lastly, we need to add the role ARN to "instance profiles" in the databricks admin console.
# MAGIC   * Now we can launch a databricks cluster with our AWS Sagemaker, which means that our cluster has access to Sagemaker resources.
# MAGIC   
# MAGIC While this is a lot to set up, remember that all the above actions have to be executed only once. After all, we only really need one MLFlow image in ECR, and authentication also only has to be set-up once

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: From Model Registry, get the URI of our production model

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

#dbutils.widgets.text("modelRegistryName","bankXGBoost")
modelRegistryName = dbutils.widgets.get("modelRegistryName")

# COMMAND ----------

def getProdModelURI(modelRegistryName):
  models = client.search_model_versions("name='%s'" % modelRegistryName)
  source = [model for model in models if model.current_stage == "Production"][0].source
  return source

modelURI = getProdModelURI(modelRegistryName)
# latest_prod_model_detail = client.get_latest_versions(model_name, stages=['Production'])[0]

# COMMAND ----------

modelURI

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Deploy our model to Sagemaker

# COMMAND ----------

# MAGIC %md
# MAGIC In the cell below we define what we want to call our Sagemaker app, and we get the mlflow image that has been registered to AWS ECR. We use mode "replace" so that we can overwrite our model in subsequent iterations.

# COMMAND ----------

app_name = "xgboostBank"
aws_account_id = "<insert account ID here!>"
aws_region = "<insert AWS ECR region here!>"
repository_name = "<insert ECR repository name here!>"
img_tag = "<insert tage here!>"

# You can also just copy paste the image url from image URI column in ECR UI
image_url = aws_account_id + ".dkr.ecr." + aws_region + ".amazonaws.com/" + repository_name ":" + img_tag

import mlflow.sagemaker as mfs
mfs.deploy(app_name=app_name, model_uri=modelURI, image_url = image_url, region_name="eu-west-1", mode="replace")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Querying our deployed model

# COMMAND ----------

df = spark.sql("select * from max_db.bank_marketing_train_set")

# COMMAND ----------

train = df.toPandas()
train_x = train.drop(["label"], axis=1)
sample = train_x.iloc[[0]]
#sample = train_x.iloc[:, :]
sample_json = sample.to_json(orient="split")

# COMMAND ----------

sample_json

# COMMAND ----------

import json
import boto3
def query_endpoint_example(app_name, input_json):
  print("Sending batch prediction request with inputs: {}".format(input_json))
  client = boto3.session.Session().client("sagemaker-runtime", "eu-west-1")
  
  response = client.invoke_endpoint(
      EndpointName=app_name,
      Body=input_json,
      ContentType='application/json; format=pandas-split',
  )
  preds = response['Body'].read().decode("ascii")
  preds = json.loads(preds)
  print("Received response: {}".format(preds))
  return preds

import pandas as pd
#input_df = pd.DataFrame([query_input])
#input_json = input_df.to_json(orient='split')

prediction1 = query_endpoint_example(app_name=app_name, input_json=sample_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Clean up our Sagemaker deployment
# MAGIC It is important to delete your Sagemaker deployment when you no longer use it, as it makes use of permanently running EC2 instances which will incur costs. For more information see [this link](https://aws.amazon.com/sagemaker/pricing/).

# COMMAND ----------

mfs.delete(app_name=app_name, region_name="eu-west-1", archive=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extra: A note on scheduling with a sagemaker deployment
# MAGIC * Scheduling can be done similar to notebook 3.1, using the Databricks scheduler. Once a deployment has been done, all you need to run is the Step 3 section.
