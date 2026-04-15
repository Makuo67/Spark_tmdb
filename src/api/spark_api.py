import os
import logging
from typing import Iterator, Dict, Any, List, Optional
import requests
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import *


load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
DEFAULT_MOVIE_IDS: List[int] = [
    299534, 19995, 140607, 299536, 597, 135397, 420818, 24428, 168259, 99861, 284054, 12445, 181808, 330457,
    351286,
    109445, 321612, 260513, 476161, 5, 1180681, 9741, 7183, 18, 2069, 1035803, 1035806, 1024546, 1571, 1572, 1573,
    761892,
    46122, 562, 5172, 774714, 63, 59967, 278086, 1507910, 2122, 2124, 9292, 1113682, 32855, 39514, 95, 30815, 3172,
    718949,
    241254, 9319, 136296, 3179, 9333, 50298, 77948, 8324, 8838, 531593, 9356, 17043, 4244, 480404, 280217, 9882,
    1414301,
    9374, 2207, 38560, 1637024, 24226, 10403, 163, 686245, 680, 137896, 234158, 486068, 186, 187, 454330, 189,
    23742, 12479,
    85693, 986824, 83666, 118483, 169173, 20694, 395990, 4824, 894169, 14043, 181471, 384737, 285923, 82150, 745,
    8944,
    872177, 504562, 641790, 9471, 18176, 584962, 7944, 921353, 1645833, 921355, 921360, 9494, 326425, 763164,
    1265440,
    7457, 146216, 542508, 883502, 139567, 32047, 651571, 345915, 479040, 11074, 11593, 345934, 84305, 1528146,
    916821,
    47964, 7518, 9567, 10592, 31586, 1460067, 30565, 12647, 381288, 360295, 126314, 829799, 787310, 72559, 843633,
    9586,
    864116, 25975, 693113, 826749, 76163, 135051, 918, 1533851, 145308, 681887, 450465, 552865, 43939, 714666,
    1049516,
    9644, 70586, 27578, 12220, 410554, 132542, 1146302, 766907, 153538, 31683, 742341, 28614, 1992, 307663, 75736,
    253414,
    1127399, 2026, 63472, 359412, 19959, 536056, 724989, 290304, 5, 66566, 22538, 67083, 1588237, 278542, 62488, 24,
    754721,
    1502241, 761892, 99368, 185896, 46122, 374317, 20013, 1173040, 539199, 1261119, 480834, 35907, 124998, 342091,
    79, 1623134,
    1228384, 1341540, 281702, 68718, 114287, 1071215, 58492, 1058940, 962192, 245906, 97430, 455319, 533658, 1690,
    1691, 458399,
    171168, 986277, 680, 92850, 1567925, 184, 187, 20668, 1598142, 1002181, 1110728, 986824, 1457866, 414419, 13025,
    285923, 63206, 594158, 413422, 1418478, 241, 1145586, 755, 443129, 515834, 36606, 1645833, 56591, 199951,
    225554, 8982,
    28447, 166183, 333106, 224562, 61752, 44345, 289083, 319, 12095, 16194, 44535, 144708, 13637, 1445188, 101204,
    466272,
    273248, 540003, 1242980, 1460067, 1088359, 631143, 1005428, 393076, 1629557, 288122, 1145722, 833916, 8068, 393,
    56224,
    19361, 149922, 396194, 1310632, 20910, 82865, 399794, 1065395, 264117, 1146302, 1010623, 1991, 1992, 339403,
    1390028,
    9678, 12241, 353746, 102868, 161239, 19416, 85984, 16869, 10213, 500, 1569780, 13300, 599031, 854521, 464890,
    507
]

BASE_URL = "https://api.themoviedb.org/3/movie/"
TIMEOUT = 15

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
        for attempt in range(5):
            try:
                r = session.get(url, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning(
                    f"[{movie_id}] Attempt {attempt+1}/5 failed: {e}")
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
