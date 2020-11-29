# Databricks notebook source
# MAGIC %md ### Reading GQA JSON files
# MAGIC 
# MAGIC This is the old code attempt at reading the JSON files. The original JSON files are probably not formatted properly/not proper JSON files, which makes Spark fail in reading them when trying to figure out the schema during reading time. We need to read the files using the pythonic approach and writing them back to files, as the resulting files might be well-formated (hopefully). However, this is not necessary at the moment, as the pythonic way works as well.

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/shared_uploads/scenegraph_motifs/")

# COMMAND ----------

# Load the training data:
######## throws an error:
# Reload the page and try again. If the error persists, contact support. Reference error code: 73f0fdda29f9485aaa2ec67c7e881e68
train_scene_data = spark.read.text("dbfs:/FileStore/shared_uploads/dali@cs.umu.se/train_sceneGraphs.json")

# COMMAND ----------

# MAGIC %scala
# MAGIC val tr_sc = sc.textFile("dbfs:/FileStore/shared_uploads/dali@cs.umu.se/train_sceneGraphs.json")

# COMMAND ----------

# MAGIC %scala
# MAGIC tr_sc.

# COMMAND ----------

train_scene_data.head(10)

# COMMAND ----------

# MAGIC %scala
# MAGIC val val_scene_data = spark.read.json("dbfs:/FileStore/shared_uploads/scenegraph_motifs/val_sceneGraphs.json")

# COMMAND ----------

# still takes too long and fails
val_scene_data = sqlContext.read.json("dbfs:/FileStore/shared_uploads/scenegraph_motifs/val_sceneGraphs.json")

# COMMAND ----------

val_scene_data.display()

# COMMAND ----------

val_scene_data.printSchema()