"""ETL pipeline: Extract nested JSON, clean data, output Parquet."""
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
import logging
from src.api.spark_api import load_data_spark
from src.config import NULL_PLACEHOLDERS, DROPNA_THRESH
# ----------------------------
# DROP IRRELEVANT COLUMNS
# ----------------------------

logger = logging.getLogger(__name__)


# ----------------------------
# EXTRACT JSON FIELDS
# ----------------------------
def extract_collection(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "belongs_to_collection",
        F.when(
            F.col("belongs_to_collection").isNotNull(),
            F.col("belongs_to_collection.name")
        ).otherwise(None)
    )


def extract_names_array(df: DataFrame, col_name: str, new_col: str) -> DataFrame:
    """
    Extract 'name' field from array<struct> and join with '|'
    """
    return (
        df.withColumn(
            new_col,
            F.when(
                F.col(col_name).isNull(),
                F.lit(None)
            ).otherwise(
                F.concat_ws("|", F.expr(f"transform({col_name}, x -> x.name)"))
            )
        )
    )


def extract_spoken_languages(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "spoken_languages",
        F.when(
            F.col("spoken_languages").isNotNull(),
            F.concat_ws("|", F.expr(
                "transform(spoken_languages, x -> x.name)"))
        )
    )


# ----------------------------
# CREDITS PROCESSING
# ----------------------------
def extract_cast(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "cast",
        F.concat_ws("|", F.expr("transform(credits.cast, x -> x.name)"))
    ).withColumn(
        "cast_size",
        F.size(F.col("credits.cast"))
    )


def extract_crew(df: DataFrame) -> DataFrame:

    return (
        df.withColumn("crew_size", F.size("credits.crew"))
          .withColumn(
              "director",
              F.expr(
                  "try_element_at(filter(credits.crew, x -> x.job = 'Director'), 1).name")
        )
    )


# ----------------------------
# TYPE CASTING
# ----------------------------
def cast_types(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("budget", F.col("budget").cast("double"))
          .withColumn("revenue", F.col("revenue").cast("double"))
          .withColumn("popularity", F.col("popularity").cast("double"))
          .withColumn("id", F.col("id").cast("int"))
          .withColumn("release_date",  F.to_date(
              F.when(F.col("release_date") == "", None)
              .otherwise(F.col("release_date"))
          ))
    )


# ----------------------------
# HANDLE INVALID VALUES
# ----------------------------
def handle_invalid_values(df: DataFrame) -> DataFrame:
    return (
        df
        # Replace zeros with null
        .withColumn("budget", F.when(F.col("budget") == 0, None).otherwise(F.col("budget")))
        .withColumn("revenue", F.when(F.col("revenue") == 0, None).otherwise(F.col("revenue")))
        .withColumn("runtime", F.when(F.col("runtime") == 0, None).otherwise(F.col("runtime")))

        # Convert to millions
        .withColumn("budget_musd", F.col("budget") / 1_000_000)
        .withColumn("revenue_musd", F.col("revenue") / 1_000_000)

        # Handle vote_count edge case
        .withColumn(
            "vote_average",
            F.when(
                (F.col("vote_count") == 0) | F.col("vote_count").isNull(),
                None
            ).otherwise(F.col("vote_average"))
        )

        # Clean text placeholders
        .withColumn(
            "overview",
            F.when(F.col("overview").isin(NULL_PLACEHOLDERS), None).otherwise(F.col("overview"))
        )
        .withColumn(
            "tagline",
            F.when(F.col("tagline").isin(NULL_PLACEHOLDERS), None).otherwise(F.col("tagline"))
        )
    )


# ----------------------------
# FILTERING
# ----------------------------
def filter_rows(df: DataFrame) -> DataFrame:
    # Remove duplicates
    df = df.dropDuplicates(["id"])

    # Drop rows with missing id or title
    df = df.filter(F.col("id").isNotNull() & F.col("title").isNotNull())

    # Keep only released movies
    df = df.filter(F.col("status") == "Released").drop("status")

    # Keep rows with at least DROPNA_THRESH non-null columns
    df = df.na.drop(thresh=DROPNA_THRESH)

    return df


# ----------------------------
# REORDER COLUMNS
# ----------------------------
logger = logging.getLogger(__name__)


def reorder_columns(df: DataFrame) -> DataFrame:
    ordered_cols = [
        'id', 'title', 'tagline', 'release_date', 'genres',
        'belongs_to_collection', 'original_language',
        'budget_musd', 'revenue_musd',
        'production_companies', 'production_countries',
        'vote_count', 'vote_average', 'popularity', 'runtime',
        'overview', 'spoken_languages', 'poster_path',
        'cast', 'cast_size', 'director', 'crew_size'
    ]

    logger.info(f"DF columns: {df.columns}")

    # DEBUG: full schema (VERY useful for nested issues)
    logger.info(f"DF schema:\n{df._jdf.schema().treeString()}")
    existing_cols = [c for c in ordered_cols if c in df.columns]
    logger.info(f"Selected columns: {existing_cols}")
    return df.select(*existing_cols)


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def preprocess(df: DataFrame) -> DataFrame:
    """
    Full preprocessing pipeline.
    Each step is isolated and composable.
    """
    df = df.withColumnRenamed("genres", "genres_raw")
    df = extract_names_array(df, "genres_raw", "genres")

    df = df.withColumnRenamed("production_companies",
                              "production_companies_raw")
    df = extract_names_array(
        df, "production_companies_raw", "production_companies")
    # Extract nested JSON
    df = extract_collection(df)
    df = extract_spoken_languages(df)

    # Credits
    df = extract_cast(df)
    df = extract_crew(df)

    # Types & cleaning
    df = cast_types(df)
    df = handle_invalid_values(df)

    # Filtering
    df = filter_rows(df)

    # Final structure
    df = reorder_columns(df)

    return df


def write_parquet_safe(df: DataFrame, output_path: str) -> None:
    """
    Safely writes Spark DataFrame to Parquet with error handling.
    """

    try:
        logger.info(f"Starting Parquet write to: {output_path}")

        (
            df.write
            .mode("overwrite")
            .option("compression", "gzip")
            .parquet(output_path)
        )

        logger.info(f"Parquet write completed successfully: {output_path}")

    except Exception as e:
        logger.error(
            f"Parquet write failed at {output_path}: {str(e)}",
            exc_info=True
        )
        raise RuntimeError(f"Spark Parquet write failed: {output_path}") from e


if __name__ == "__main__":
    df = load_data_spark()
    df = preprocess(df)
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_PATH = BASE_DIR / "/Spark_tmdb/data"
    write_parquet_safe(df, str(OUTPUT_PATH))
