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

import uuid
uniqueId = uuid.uuid4()
spark.conf.set("com.databricks.tmp.uniqueid", str(uniqueId))

# COMMAND ----------

username = spark.conf.get("com.databricks.tmp.username")
userhome = spark.conf.get("com.databricks.tmp.userhome")
uniqueid = spark.conf.get("com.databricks.tmp.uniqueid")