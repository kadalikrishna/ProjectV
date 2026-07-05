"""Initialize and seed the HDI Predictor SQLite database."""

import os
from pathlib import Path

import pandas as pd

from database import (
    DEFAULT_MODEL_NAME,
    get_connection,
    get_or_create_default_user,
    initialize_database,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "hdi_model.pkl"


def seed_dataset_metadata(connection):
    """Store dataset metadata in the dataset table."""
    if not DATASET_PATH.exists():
        return

    df = pd.read_csv(DATASET_PATH)
    connection.execute(
        """
        INSERT INTO dataset (
            dataset_name, source, total_rows, total_columns
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(dataset_name) DO UPDATE SET
            source = excluded.source,
            total_rows = excluded.total_rows,
            total_columns = excluded.total_columns
        """,
        (
            "Human Development Index - Full.csv",
            "UNDP Human Development Index dataset",
            int(df.shape[0]),
            int(df.shape[1]),
        ),
    )


def seed_model_metadata(connection):
    """Store the active ML model metadata."""
    connection.execute(
        """
        INSERT INTO ml_model (
            model_name, algorithm_used, accuracy_score, r2_score, model_file_path
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET
            algorithm_used = excluded.algorithm_used,
            accuracy_score = excluded.accuracy_score,
            r2_score = excluded.r2_score,
            model_file_path = excluded.model_file_path
        """,
        (
            DEFAULT_MODEL_NAME,
            "Linear Regression",
            0.9582,
            0.9582,
            os.path.relpath(MODEL_PATH, BASE_DIR),
        ),
    )


def seed_countries(connection):
    """Load countries and regions from dataset.csv."""
    if not DATASET_PATH.exists():
        return

    df = pd.read_csv(DATASET_PATH)
    if "Country" not in df.columns:
        return

    region_column = "UNDP Developing Regions"
    countries = []
    for _, row in df.iterrows():
        country_name = row.get("Country")
        if pd.isna(country_name) or not str(country_name).strip():
            continue
        region = row.get(region_column) if region_column in df.columns else None
        if pd.isna(region):
            region = None
        countries.append((str(country_name).strip(), region, None))

    connection.executemany(
        """
        INSERT INTO country (country_name, region, population)
        VALUES (?, ?, ?)
        ON CONFLICT(country_name) DO UPDATE SET
            region = excluded.region,
            population = excluded.population
        """,
        countries,
    )


def main():
    """Create schema and seed default ERD records."""
    initialize_database()
    with get_connection() as connection:
        get_or_create_default_user(connection)
        seed_dataset_metadata(connection)
        seed_model_metadata(connection)
        seed_countries(connection)
        connection.commit()

        counts = {
            table: connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()[
                "count"
            ]
            for table in [
                "user",
                "session",
                "country",
                "dataset",
                "ml_model",
                "hdi_input_data",
                "hdi_prediction",
                "visualization_report",
            ]
        }

    print(f"Database initialized at {BASE_DIR / 'hdi_predictor.db'}")
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
