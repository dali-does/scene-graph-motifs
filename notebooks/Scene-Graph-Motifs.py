# Databricks notebook source
# MAGIC %md # Exploring Motifs in Scene Graph Data
# MAGIC 
# MAGIC This project aims to explore the scene graphs in the General Question Answering (GQA) dataset [1]. The motifs of the ground truth data will be analysed, and possibly compared to predictions by pretrained models.
# MAGIC 
# MAGIC [1]: https://cs.stanford.edu/people/dorarad/gqa/index.html

# COMMAND ----------

# MAGIC %md ##Graph structure
# MAGIC 
# MAGIC - We want to extract the names of objects we see in the images ans use their id's as vertices (create a object_name2id dictionary)
# MAGIC - For the same object category, we will have multiple id's, but the category is represented by only one vertex in the graph (coarse-graph representation);
# MAGIC - The edge properties are the names of the relations in each .json file; we extract them and store in triples (?) and create edge_name2id.

# COMMAND ----------

# MAGIC %md ## Loading data
# MAGIC 
# MAGIC We read the scene graph data as JSON files. Below is the example JSON object given by the GQA website, for scene graph 2407890.

# COMMAND ----------

sc = spark.sparkContext
# Had to change weather 'none' to '"none"' for the string to parse
json_example_str = '{"2407890": {"width": 640,"height": 480,"location": "living room","weather": "none","objects": {"271881": {"name": "chair","x": 220,"y": 310,"w": 50,"h": 80,"attributes": ["brown", "wooden", "small"],"relations": {"32452": {"name": "on","object": "275312"},"32452": {"name": "near","object": "279472"}}}}}}'
json_rdd = sc.parallelize([json_example_str])
example_json_df = spark.read.json(json_rdd, multiLine=True)
example_json_df.show()

# COMMAND ----------

example_json_df.first()

# COMMAND ----------

# MAGIC %md ### Reading JSON files

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

# COMMAND ----------

# MAGIC %md ### Parsing graph structure

# COMMAND ----------

from graphframes import *
import json

# COMMAND ----------

f = open("/dbfs/FileStore/shared_uploads/scenegraph_motifs/val_sceneGraphs.json")
val_scene_data = json.load(f)

# COMMAND ----------


# Pythonic way of doing it, must be converted to dataframe processing
##### Pavlo's comment: the pythonic way may be a not bad option https://stackoverflow.com/questions/39818368/convert-lines-of-json-in-rdd-to-dataframe-in-apache-spark
def json_to_vertices_edges(graph_json):
  vertices = []
  edges = []
  vertice_ids = graph_json['objects']
  for vertice_id in vertice_ids:
    vertice_obj = graph_json['objects'][vertice_id]
    name = vertice_obj['name']
    vertices.append((vertice_id, name))
    for relation in vertice_obj['relations']:
        src = vertice_id
        dst = relation['object']
        name = relation['name']
        edges.append((src, dst, name))
  return (vertices, edges)

# COMMAND ----------



# COMMAND ----------

# Pythonic way of doing it, must be converted to dataframe processing
def parse_scene_graphs(scene_graphs_json, vertice_schema, edge_schema):
  
  vertices = []
  edges = []
  
  for scene_graph_id in scene_graphs_json:
    vs, es = json_to_vertices_edges(scene_graphs_json[scene_graph_id])
    vertices += vs
    edges += es
    
  vertices = spark.createDataFrame(vertices, vertice_schema)
  edges = spark.createDataFrame(edges, edge_schema)
  
  return GraphFrame(vertices, edges)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType
# Create schemas for scene graphs
vertice_schema = StructType([
  StructField("id", StringType(), False), StructField("object_name", StringType(), False)
])
  
edge_schema = StructType([
  StructField("src", StringType(), False), StructField("dst", StringType(), False), StructField("relation_name", StringType(), False)
])

# COMMAND ----------

scene_graphs = parse_scene_graphs(example_json, vertice_schema, edge_schema)

# COMMAND ----------

display(scene_graphs.vertices)

# COMMAND ----------

display(scene_graphs.edges)

# COMMAND ----------



# COMMAND ----------

