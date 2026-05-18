# TMDB Spark Pipeline

PySpark ETL pipeline that ingests ~500 movies from the TMDB API, processes nested JSON into a clean Parquet dataset, and computes box-office KPIs, rankings, and aggregations.

---

## Architecture

```
TMDB REST API
      │
      ▼
spark_api.py          ← parallel fetch via RDD.mapPartitions, Pydantic validation
      │
      ▼  (raw Spark DataFrame)
pre_process.py        ← extract nested structs, cast types, null/placeholder cleaning,
      │                  thresh-based row filter, Parquet write
      ▼  (data/*.parquet)
analytics.py          ← KPI computation, dense_rank windows, aggregations
      │
      ▼
main.py               ← orchestration entry point
notebook/             ← EDA and visualizations (matplotlib / seaborn)
```

---

## Repository Layout

```
Spark_tmdb/
├── src/
│   ├── config.py              # all constants: thresholds, API config, null placeholders
│   ├── api/
│   │   └── spark_api.py       # TMDB fetch, Pydantic validation, Spark schema
│   └── helpers/
│       ├── pre_process.py     # ETL pipeline (extract → clean → filter → write)
│       ├── analytics.py       # KPIs, ranking windows, franchise/director stats
│       └── main.py            # entry point: loads Parquet, runs analytics
├── notebook/
│   └── visualization.ipynb    # revenue trends, genre distributions, scatter plots
├── data/                      # Parquet output (gzip-compressed, single partition)
├── requirements.txt
└── .env                       # TMDB_API_KEY (not committed)
```

---

## Setup

**Prerequisites:** Python 3.10+, Java 11+ (required by PySpark).

```bash
git clone <repo>
cd Spark_tmdb
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
TMDB_API_KEY=your_key_here
```

---

## Running the Pipeline

**Step 1 — Ingest and build Parquet:**

```bash
cd Spark_tmdb   # project root
python -m src.helpers.pre_process
```

This fetches movies in parallel (4 RDD partitions), validates each record with Pydantic, extracts nested structs (genres, cast, crew, collection), applies cleaning rules, and writes `data/*.parquet`.

**Step 2 — Run analytics:**

```bash
cd src/helpers
python main.py
```

**Step 3 — Explore visuals:**

```bash
jupyter notebook notebook/visualization.ipynb
```

---

## ETL Details

### Ingestion (`spark_api.py`)

- `sc.parallelize(movie_ids, numSlices=4)` distributes IDs across executors.
- Each partition opens one `requests.Session` and fetches `/movie/{id}?append_to_response=credits`.
- Retry policy: up to 5 attempts with exponential backoff (`2^attempt` seconds). HTTP 429 responses respect the `Retry-After` header before retrying.
- Records are validated against a Pydantic `Movie` model; invalid records are logged and dropped.

### Transformation (`pre_process.py`)

| Step                   | What happens                                                                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Struct extraction      | `genres`, `production_companies` unnested from `array<struct>` → `\|`-delimited strings                                                                   |
| Credits extraction     | `cast` → pipe-delimited names; `director` extracted from crew where `job = 'Director'`                                                                    |
| Type casting           | `budget`, `revenue` → `double`; `release_date` → `date`; `id` → `int`                                                                                     |
| Invalid value handling | Zero-valued `budget`/`revenue`/`runtime` → `null`; `vote_average` nulled when `vote_count = 0`                                                            |
| Placeholder nulling    | `overview`, `tagline` strings matching `["No Data", "N/A", "None", "null", "Unknown", ""]` → `null`                                                       |
| Row filtering          | Deduplicate on `id`; require non-null `id` and `title`; keep `status = Released`; drop rows with fewer than 10 non-null columns (`df.na.drop(thresh=10)`) |
| Units                  | `budget_musd`, `revenue_musd` stored in millions USD                                                                                                      |

### Output schema (ordered columns)

```
id, title, tagline, release_date, genres, belongs_to_collection,
original_language, budget_musd, revenue_musd, production_companies,
production_countries, vote_count, vote_average, popularity, runtime,
overview, spoken_languages, poster_path, cast, cast_size, director, crew_size
```

---

## Analytics (`analytics.py`)

All rankings use `dense_rank()` over a global window (no partition) ordered by the target metric.

### KPIs — `best_worst_movies()`

| Key               | Metric              | Filter              |
| ----------------- | ------------------- | ------------------- |
| `highest_revenue` | `revenue_musd` desc | —                   |
| `highest_budget`  | `budget_musd` desc  | —                   |
| `highest_profit`  | `profit` desc       | —                   |
| `lowest_profit`   | `profit` asc        | —                   |
| `highest_roi`     | `roi` desc          | `budget_musd >= 10` |
| `lowest_roi`      | `roi` asc           | `budget_musd >= 10` |
| `most_voted`      | `vote_count` desc   | —                   |
| `highest_rated`   | `vote_average` desc | `vote_count >= 10`  |
| `lowest_rated`    | `vote_average` asc  | `vote_count >= 10`  |
| `most_popular`    | `popularity` desc   | —                   |

`profit = revenue_musd - budget_musd`  
`roi = revenue_musd / budget_musd` (only where `budget_musd > 0`)

Budget and vote-count thresholds are configurable via `src/config.py` (`MIN_BUDGET_MUSD`, `MIN_VOTE_COUNT`).

### Aggregations

- **`franchise_vs_standalone()`** — groups by `belongs_to_collection IS NOT NULL`, computes mean revenue, median ROI (percentile_approx), mean budget, mean popularity, mean rating.
- **`top_franchises()`** — groups by collection name, sums/averages financials, ordered by total revenue.
- **`top_directors()`** — groups by director, totals revenue and counts titles, ordered by total revenue.

### Searches — `search_movies()`

- Bruce Willis films tagged both `Science Fiction` and `Action`, ranked by rating.
- Quentin Tarantino × Uma Thurman collaborations, ranked by runtime.

---

## Configuration (`src/config.py`)

All magic numbers live here. Change them once; every module picks them up.

| Constant            | Default  | Purpose                                 |
| ------------------- | -------- | --------------------------------------- |
| `MIN_BUDGET_MUSD`   | `10.0`   | ROI ranking budget floor                |
| `MIN_VOTE_COUNT`    | `10`     | Rating ranking vote floor               |
| `DROPNA_THRESH`     | `10`     | Minimum non-null columns per row        |
| `TIMEOUT`           | `15`     | HTTP request timeout (seconds)          |
| `MAX_RETRIES`       | `5`      | API retry attempts                      |
| `NULL_PLACEHOLDERS` | see file | Strings coerced to null during cleaning |

---

## Key Design Decisions

- **Pydantic validation at ingest boundary.** Structural errors are caught before data enters Spark, keeping the DataFrame schema clean and avoiding silent nulls from schema mismatches.
- **`mapPartitions` over `map`.** One HTTP session per partition rather than per record reduces connection overhead.
- **`dense_rank()` global window for KPIs.** Rankings are over the full dataset; no partition column is passed to `rank_within()`. Metrics with a low-sample bias (ROI, ratings) apply explicit filters before ranking.
- **Parquet + gzip for storage.** Columnar format suits the aggregation-heavy query pattern; gzip gives good compression ratios for this dataset size without needing Snappy native libs.
- **Thresholds and null placeholders in config.** Avoids scattered literals across ETL and analytics code, making threshold tuning a one-line change.

---

## Limitations

- Dataset is ~500 hand-curated IDs. Conclusions about "industry trends" are directional, not statistically representative.
- `append_to_response=credits` is a single API call per film; rate-limit headroom on a free TMDB key is ~40 req/s, constraining parallelism.
- Single-node Spark (local mode). The pipeline is structured for distributed execution but has not been tested on a cluster.
