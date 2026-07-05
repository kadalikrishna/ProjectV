"""Initialize and seed the HDI Predictor SQLite database."""

from database import DATABASE_PATH, ensure_seed_data, get_connection


def main():
    """Create schema and seed default ERD records."""
    ensure_seed_data()
    with get_connection() as connection:
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

    print(f"Database initialized at {DATABASE_PATH}")
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
