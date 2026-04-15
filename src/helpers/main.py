from pyspark.sql import SparkSession
from analytics import *

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("../data")

# KPIs
kpis = best_worst_movies(df)

kpis["highest_revenue"].show()
kpis["highest_roi"].show()

# Searches
queries = search_movies(df)
queries["bruce_willis_sci_fi"].show()

# Aggregations
franchise_stats = franchise_vs_standalone(df)
franchise_stats.show()

top_franchise = top_franchises(df)
top_franchise.show()

top_director = top_directors(df)
top_director.show()
