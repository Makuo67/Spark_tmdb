# Spark_tmdb Analysis Report

## Overview

PySpark pipeline analyzing TMDB movies dataset (~500 films). Covers data ingestion, ETL, KPIs, and visualizations.

## Key Insights

- **Franchises outperform**: Higher revenue, popularity vs standalone.
- **Financial KPIs**: Profit = revenue - budget; ROI = revenue/budget (budget>0).
- **Data Quality**: ~20% movies have budget=0 (nullified); only released films.
- **Top Analytics**: Highest ROI/revenue rankings, director/franchise aggregations, searches (e.g., Bruce Willis sci-fi).
- **Trends**: Yearly box office via visualizations.

## Methodology

1. **Ingestion** (`spark_api.py`): Parallel API fetch → Pydantic-validated Spark DF.
2. **ETL** (`pre_process.py`): Extract JSON (genres/cast/crew), clean (null zeros, filter duplicates/released), type cast, Parquet output (`src/data/*.parquet`).
3. **Analytics** (`analytics.py`): KPIs (ROI/profit), rankings, franchise/director stats, queries.
4. **Execution** (`main.py`): Load Parquet → run analytics.
5. **Visuals** (`visualization.ipynb`): Revenue/budget scatter, ROI by genre boxplot, trends.

Processed: `data/*.parquet`.

## Conclusions

- Scalable Spark pipeline handles nested JSON/messy data.
- Actionable: Franchises + top directors = revenue drivers.
- Ready for production: Error-handling, logging, compression.

## Quick Start

```bash
cd Spark_tmdb/src/helpers
python main.py  # Run analytics
jupyter notebook ../notebook/visualization.ipynb  # Plots
```
