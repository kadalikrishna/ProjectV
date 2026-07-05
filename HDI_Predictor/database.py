"""SQLite helpers for the HDI Predictor ERD tables."""

import os
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "hdi_predictor.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"
DEFAULT_USER_EMAIL = "guest@hdi-predictor.local"
DEFAULT_MODEL_NAME = "HDI Linear Regression Model"


def get_connection():
    """Create a SQLite connection with foreign keys and row dictionaries."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create all ERD tables from schema.sql."""
    with get_connection() as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.commit()


def get_or_create_default_user(connection):
    """Return the default guest user id."""
    row = connection.execute(
        "SELECT user_id FROM user WHERE email = ?", (DEFAULT_USER_EMAIL,)
    ).fetchone()
    if row:
        return row["user_id"]

    cursor = connection.execute(
        """
        INSERT INTO user (name, email, role)
        VALUES (?, ?, ?)
        """,
        ("Guest User", DEFAULT_USER_EMAIL, "guest"),
    )
    return cursor.lastrowid


def create_session(connection, user_id, status="active"):
    """Create a user session row and return its id."""
    cursor = connection.execute(
        """
        INSERT INTO session (user_id, status)
        VALUES (?, ?)
        """,
        (user_id, status),
    )
    return cursor.lastrowid


def get_default_model_id(connection):
    """Return the seeded model id, creating it if needed."""
    row = connection.execute(
        "SELECT model_id FROM ml_model WHERE model_name = ?", (DEFAULT_MODEL_NAME,)
    ).fetchone()
    if row:
        return row["model_id"]

    cursor = connection.execute(
        """
        INSERT INTO ml_model (
            model_name, algorithm_used, accuracy_score, r2_score, model_file_path
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            DEFAULT_MODEL_NAME,
            "Linear Regression",
            0.9582,
            0.9582,
            "hdi_model.pkl",
        ),
    )
    return cursor.lastrowid


def get_country_id_by_name(connection, country_name):
    """Look up a country id by name."""
    if not country_name:
        return None

    row = connection.execute(
        "SELECT country_id FROM country WHERE country_name = ?", (country_name,)
    ).fetchone()
    return row["country_id"] if row else None


def list_countries():
    """Return country names for the prediction dropdown."""
    if not os.path.exists(DATABASE_PATH):
        initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT country_name FROM country ORDER BY country_name"
        ).fetchall()
    return [row["country_name"] for row in rows]


def save_prediction_record(values, predicted_score, category, country_name=None):
    """Persist one full ERD prediction flow and return the prediction id."""
    initialize_database()
    with get_connection() as connection:
        user_id = get_or_create_default_user(connection)
        create_session(connection, user_id)
        model_id = get_default_model_id(connection)
        country_id = get_country_id_by_name(connection, country_name)

        input_cursor = connection.execute(
            """
            INSERT INTO hdi_input_data (
                user_id, country_id, life_expectancy, mean_years_schooling,
                expected_years_schooling, gni_per_capita
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, country_id, values[0], values[1], values[2], values[3]),
        )
        input_id = input_cursor.lastrowid

        prediction_cursor = connection.execute(
            """
            INSERT INTO hdi_prediction (
                input_id, model_id, predicted_hdi_score, hdi_category
            )
            VALUES (?, ?, ?, ?)
            """,
            (input_id, model_id, predicted_score, category),
        )
        prediction_id = prediction_cursor.lastrowid

        report_paths = [
            ("static/plots/heatmap.png", "Heatmap"),
            ("static/plots/correlation_matrix.png", "Correlation Matrix"),
            ("static/plots/distribution_plot.png", "Distribution Plot"),
            ("static/plots/scatter_plot.png", "Scatter Plot"),
            ("static/plots/strip_plot.png", "Strip Plot"),
        ]
        connection.executemany(
            """
            INSERT INTO visualization_report (
                prediction_id, graph_path, report_type
            )
            VALUES (?, ?, ?)
            """,
            [(prediction_id, graph_path, report_type) for graph_path, report_type in report_paths],
        )
        connection.commit()
        return prediction_id


def get_recent_predictions(limit=10):
    """Return recent prediction history with input and country details."""
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.prediction_id,
                p.predicted_hdi_score,
                p.hdi_category,
                p.prediction_time,
                i.life_expectancy,
                i.mean_years_schooling,
                i.expected_years_schooling,
                i.gni_per_capita,
                c.country_name,
                m.model_name
            FROM hdi_prediction p
            JOIN hdi_input_data i ON i.input_id = p.input_id
            JOIN ml_model m ON m.model_id = p.model_id
            LEFT JOIN country c ON c.country_id = i.country_id
            ORDER BY p.prediction_time DESC, p.prediction_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
