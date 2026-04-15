import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window


# ----------------------------
# BASE METRICS
# ----------------------------
def add_kpis(df: DataFrame) -> DataFrame:
    """Profit and ROI metrics."""
    return (
        df.withColumn("profit", F.col("revenue_musd") - F.col("budget_musd"))
          .withColumn(
              "roi",
              F.when(F.col("budget_musd") > 0,
                     F.col("revenue_musd") / F.col("budget_musd"))
        )
    )


# ----------------------------
# RANKING UTIL
# ----------------------------
def rank_within(
    df: DataFrame,
    order_col: str,
    partition_col: str | None = None,
    desc: bool = True,
    rank_col: str = "rank",
    keep_cols: list | None = None
) -> DataFrame:

    order_expr = F.col(order_col).desc() if desc else F.col(order_col).asc()

    window = (
        Window.partitionBy(partition_col).orderBy(order_expr)
        if partition_col else
        Window.orderBy(order_expr)
    )

    df = df.withColumn(rank_col, F.dense_rank().over(window))

    return df.select(*keep_cols) if keep_cols else df


# ----------------------------
# KPI EXTRACTIONS
# ----------------------------
def best_worst_movies(df: DataFrame) -> dict:
    df = add_kpis(df)

    return {
        "highest_revenue": rank_within(df, "revenue_musd", keep_cols=["title", "genres", "revenue_musd", "rank"]).limit(10),
        "highest_budget": rank_within(df, "budget_musd", keep_cols=["title", "genres", "budget_musd", "rank"]).limit(10),
        "highest_profit": rank_within(df, "profit", keep_cols=["title", "genres", "profit", "rank"]).limit(10),
        "lowest_profit": rank_within(df, "profit", keep_cols=["title", "genres", "profit", "rank"], desc=False).limit(10),

        "highest_roi": rank_within(
            df.filter(F.col("budget_musd") >= 10), "title", "roi", keep_cols=["title", "genres", "roi", "rank"]
        ).limit(10),

        "lowest_roi": rank_within(
            df.filter(F.col("budget_musd") >= 10), "title", "roi", keep_cols=["title", "genres", "roi", "rank"], desc=False
        ).limit(10),

        "most_voted": rank_within(df, "title", "vote_count", keep_cols=["title", "genres", "vote_count", "rank"]).limit(10),

        "highest_rated": rank_within(
            df.filter(F.col("vote_count") >= 10), "title", "vote_average", keep_cols=["title", "genres", "vote_average", "rank"]
        ).limit(10),

        "lowest_rated": rank_within(
            df.filter(F.col("vote_count") >= 10), "title",
            "vote_average", keep_cols=["title", "genres", "vote_average", "rank"],
            desc=False
        ).limit(10),

        "most_popular": rank_within(df, "title", "popularity", keep_cols=["title", "genres", "popularity", "rank"]).limit(10),
    }


# ----------------------------
# SEARCH QUERIES
# ----------------------------
def search_movies(df: DataFrame) -> dict:

    df = (
        df.withColumn("genres_array", F.split("genres", "\\|"))
          .withColumn("cast_array", F.split("cast", "\\|"))
    )

    sci_fi_action = (
        df.filter(
            F.array_contains(F.col("genres_array"), "Science Fiction") &
            F.array_contains(F.col("genres_array"), "Action") &
            F.array_contains(F.col("cast_array"), "Bruce Willis")
        )
        .orderBy(F.col("vote_average").desc())
    )

    tarantino_uma = (
        df.filter(
            F.array_contains(F.col("cast_array"), "Uma Thurman") &
            (F.col("director") == "Quentin Tarantino")
        )
        .orderBy(F.col("runtime").asc())
    ).select(
        "title", "director", "cast", "runtime", "vote_average"
    )

    return {
        "bruce_willis_sci_fi": sci_fi_action,
        "tarantino_uma": tarantino_uma,
    }


# ----------------------------
# FRANCHISE VS STANDALONE
# ----------------------------
def franchise_vs_standalone(df: DataFrame) -> DataFrame:
    df = add_kpis(df)

    return (
        df.withColumn(
            "is_franchise",
            F.when(F.col("belongs_to_collection").isNotNull(), 1).otherwise(0)
        )
        .groupBy("is_franchise")
        .agg(
            F.mean("revenue_musd").alias("mean_revenue"),
            F.expr("percentile_approx(roi, 0.5)").alias("median_roi"),
            F.mean("budget_musd").alias("mean_budget"),
            F.mean("popularity").alias("mean_popularity"),
            F.mean("vote_average").alias("mean_rating")
        )
    )


# ----------------------------
# FRANCHISE SUCCESS
# ----------------------------
def top_franchises(df: DataFrame) -> DataFrame:
    df = add_kpis(df)

    return (
        df.filter(F.col("belongs_to_collection").isNotNull())
        .groupBy("belongs_to_collection")
        .agg(
            F.count("*").alias("movie_count"),
            F.sum("budget_musd").alias("total_budget"),
            F.mean("budget_musd").alias("mean_budget"),
            F.sum("revenue_musd").alias("total_revenue"),
            F.mean("revenue_musd").alias("mean_revenue"),
            F.mean("vote_average").alias("mean_rating")
        )
        .orderBy(F.col("total_revenue").desc())
    )


# ----------------------------
# DIRECTOR SUCCESS
# ----------------------------
def top_directors(df: DataFrame) -> DataFrame:
    return (
        df.filter(F.col("director").isNotNull())
        .groupBy("director")
        .agg(
            F.count("*").alias("movie_count"),
            F.sum("revenue_musd").alias("total_revenue"),
            F.mean("vote_average").alias("mean_rating")
        )
        .orderBy(F.col("total_revenue").desc())
    )
