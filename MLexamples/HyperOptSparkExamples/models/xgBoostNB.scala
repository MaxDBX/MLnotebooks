// Databricks notebook source
// get parameters + data
val max_depth: Int = dbutils.widgets.get("max_depth").toInt
val trainData = spark.sql("select * from globalTempTrainData")
val testData = spark.sql("select * from globalTempTestData")

// train model
val xgbClassifier = (new XGBoostClassifier()...)
val xgbClassificationModel = xgbClassifier.fit(denseTrainData)
val evaluator = (new MulticlassClassificationEvaluator()...)

// get loss
val loss = 1 - evaluator.evaluate(xgbClassificationModel.transform(testData))

// save model and return loss
xgbClassificationModel.nativeBooster.saveModel(dbfsPath)
dbutils.notebook.exit(loss.toString)