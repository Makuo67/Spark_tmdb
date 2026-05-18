import os
import time
import logging
from typing import Iterator, Dict, Any, List, Optional
import requests
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from src.config import BASE_URL, TIMEOUT, MAX_RETRIES, DEFAULT_MOVIE_IDS


"""TMDB API loader for PySpark. Fetches movies in parallel RDDs with validation."""
load_dotenv()


# BASE_URL, TIMEOUT, MAX_RETRIES, DEFAULT_MOVIE_IDS imported from src.config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# PYDANTIC MODEL
# ----------------------------


# ----------------------------
# NESTED MODELS
# ----------------------------
class Genre(BaseModel):
    id: int
    name: str


class ProductionCompany(BaseModel):
    id: int
    name: str
    logo_path: Optional[str]
    origin_country: Optional[str]


class ProductionCountry(BaseModel):
    iso_3166_1: str
    name: str


class SpokenLanguage(BaseModel):
    english_name: str
    iso_639_1: str
    name: str


class Collection(BaseModel):
    id: int
    name: str
    poster_path: Optional[str]
    backdrop_path: Optional[str]


# ----------------------------
# CREDITS MODELS
# ----------------------------
class CastMember(BaseModel):
    id: int
    name: str
    character: Optional[str]
    gender: Optional[int]
    order: Optional[int]
    popularity: Optional[float]
    profile_path: Optional[str]


class CrewMember(BaseModel):
    id: int
    name: str
    job: Optional[str]
    department: Optional[str]
    gender: Optional[int]
    popularity: Optional[float]
    profile_path: Optional[str]


class Credits(BaseModel):
    cast: List[CastMember]
    crew: List[CrewMember]


# ----------------------------
# MAIN MOVIE MODEL
# ----------------------------
class Movie(BaseModel):
    id: int
    title: str

    # Basic info
    adult: Optional[bool]
    backdrop_path: Optional[str]
    homepage: Optional[str]
    imdb_id: Optional[str]
    original_language: Optional[str]
    original_title: Optional[str]
    overview: Optional[str]
    tagline: Optional[str]
    status: Optional[str]

    # Dates & metrics
    release_date: Optional[str]
    runtime: Optional[int]
    popularity: Optional[float]
    vote_average: Optional[float]
    vote_count: Optional[int]

    # Financials
    budget: Optional[int]
    revenue: Optional[int]

    # Media
    poster_path: Optional[str]
    video: Optional[bool]

    # Arrays
    genres: Optional[List[Genre]]
    origin_country: Optional[List[str]]
    production_companies: Optional[List[ProductionCompany]]
    production_countries: Optional[List[ProductionCountry]]
    spoken_languages: Optional[List[SpokenLanguage]]

    # Nested object
    belongs_to_collection: Optional[Collection]

    # Credits
    credits: Optional[Credits]

    class Config:
        extra = "ignore"


# ----------------------------
# SESSION FACTORY
# ----------------------------
def create_session() -> requests.Session:
    """Create HTTP session."""
    return requests.Session()


# ----------------------------
# FETCH FUNCTION
# ----------------------------
def fetch_movie(movie_id: int, api_key: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """Fetch movie + credits with basic retry."""
    def fetch(url, params):
        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(url, params=params, timeout=TIMEOUT)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"[{movie_id}] Rate limited (429). Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[{movie_id}] Attempt {attempt+1}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)
        return None

    url = f"{BASE_URL}{movie_id}"
    params = {"api_key": api_key, "append_to_response": "credits"}
    movie = fetch(url, params)

    if not movie:
        return None

    # fallback for missing credits
    if "credits" not in movie or not isinstance(movie["credits"], dict):
        credits_url = f"{BASE_URL}{movie_id}/credits"
        credits = fetch(credits_url, {"api_key": api_key})
        if not credits:
            return None
        movie["credits"] = credits

    return movie


# ----------------------------
# VALIDATION
# ----------------------------
def validate_movie(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate using Pydantic and return dict."""
    try:
        return Movie(**data).dict()
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        return None


# ----------------------------
# PARTITION PROCESSOR
# ----------------------------
def process_partition(
    movie_ids: Iterator[int],
    api_key: str
) -> Iterator[Dict[str, Any]]:
    """
    Runs inside Spark executor.
    Handles API calls per partition to reduce session overhead.
    """
    session = create_session()

    for movie_id in movie_ids:
        data = fetch_movie(movie_id, api_key, session)
        if not data:
            continue

        validated = validate_movie(data)
        if validated:
            yield validated


# ----------------------------
# MAIN SPARK LOADER
# ----------------------------
def load_data_spark(
    movie_ids: Optional[List[int]] = None,
    api_key: Optional[str] = None
):
    """Main entrypoint returning Spark DataFrame."""
    movie_ids = movie_ids or DEFAULT_MOVIE_IDS
    api_key = api_key or os.getenv("TMDB_API_KEY")

    if not api_key:
        raise ValueError("TMDB_API_KEY must be set")

    spark = SparkSession.builder \
        .appName("TMDB PySpark Loader") \
        .config("spark.sql.parquet.compression.codec", "gzip") \
        .config("spark.io.compression.codec", "lz4") \
        .config("spark.driver.extraJavaOptions", "-Dorg.xerial.snappy.use.systemlib=false") \
        .config("spark.executor.extraJavaOptions", "-Dorg.xerial.snappy.use.systemlib=false") \
        .getOrCreate()

    sc = spark.sparkContext

    # Parallelize IDs
    rdd = sc.parallelize(movie_ids, numSlices=4)

    # Distributed fetch
    result_rdd = rdd.mapPartitions(
        lambda partition: process_partition(partition, api_key)
    )

    # Schema definition for DataFrame
    schema = StructType([
        StructField("id", LongType()),
        StructField("revenue", LongType()),
        StructField("budget", LongType()),
        StructField("popularity", FloatType()),
        StructField("runtime", LongType()),
        StructField("title", StringType()),
        StructField("vote_count", LongType()),
        StructField("release_date", StringType()),
        StructField("vote_average", FloatType()),
        StructField("tagline", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("poster_path", StringType(), True),
        StructField("original_language", StringType(), True),
        StructField("status", StringType(), True),
        StructField("production_companies", ArrayType(
            StructType([
                StructField("id", LongType()),
                StructField("name", StringType()),
                StructField("origin_country", StringType())
            ])
        )),

        StructField("production_countries", ArrayType(
            StructType([
                StructField("iso_3166_1", StringType()),
                StructField("name", StringType())
            ])
        )),

        StructField("spoken_languages", ArrayType(
            StructType([
                StructField("english_name", StringType()),
                StructField("iso_639_1", StringType()),
                StructField("name", StringType())
            ])
        )),

        StructField("belongs_to_collection", StructType([
            StructField("id", LongType()),
            StructField("name", StringType()),
            StructField("poster_path", StringType()),
            StructField("backdrop_path", StringType())
        ])),

        StructField("genres", ArrayType(
            StructType([
                StructField("id", LongType()),
                StructField("name", StringType())
            ])
        )),

        StructField("credits", StructType([
            StructField("cast", ArrayType(
                StructType([
                    StructField("id", LongType()),
                    StructField("name", StringType()),
                    StructField("character", StringType())
                ])
            )),
            StructField("crew", ArrayType(
                StructType([
                    StructField("id", LongType()),
                    StructField("name", StringType()),
                    StructField("job", StringType()),
                    StructField("department", StringType())
                ])
            ))
        ]))
    ])

    # Convert credits dict -> string for Spark compatibility
    final_rdd = result_rdd.map(lambda x: {
        **x,
        "credits": x.get("credits")
    })

    df = spark.createDataFrame(final_rdd, schema=schema)
    logger.info(f"DF columns: {df.columns}")

    logger.info(f"Loaded {df.count()} movies out of {len(movie_ids)}")

    return df


# ----------------------------
# ENTRYPOINT
# ----------------------------
if __name__ == "__main__":
    df = load_data_spark()
    df.show(truncate=False)
