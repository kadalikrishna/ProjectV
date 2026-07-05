"""Flask app for predicting Human Development Index scores."""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

from database import get_recent_predictions, list_countries, save_prediction_record


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hdi_model.pkl")
FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI per Capita",
]
INPUT_LIMITS = {
    "Life Expectancy": (0, 120),
    "Mean Years of Schooling": (0, 30),
    "Expected Years of Schooling": (0, 30),
    "GNI per Capita": (0, 250000),
}


def load_model():
    """Load the trained Pickle model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "hdi_model.pkl was not found. Please run train_model.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def classify_hdi(score):
    """Convert an HDI score into the official development category."""
    if score >= 0.800:
        return "Very High Human Development"
    if score >= 0.700:
        return "High Human Development"
    if score >= 0.550:
        return "Medium Human Development"
    return "Low Human Development"


def category_message(category):
    """Return a short explanation for the predicted HDI category."""
    messages = {
        "Very High Human Development": (
            "This result suggests strong outcomes in health, education, and income."
        ),
        "High Human Development": (
            "This result suggests solid development with room for continued progress."
        ),
        "Medium Human Development": (
            "This result suggests moderate development and important improvement areas."
        ),
        "Low Human Development": (
            "This result suggests significant gaps in core development indicators."
        ),
    }
    return messages.get(category, "HDI category could not be explained.")


def validate_inputs(values):
    """Check that values stay inside realistic ranges before prediction."""
    for column_name, value in zip(FEATURE_COLUMNS, values):
        minimum, maximum = INPUT_LIMITS[column_name]
        if value < minimum or value > maximum:
            raise ValueError(
                f"{column_name} must be between {minimum:g} and {maximum:g}."
            )


@app.route("/")
def home():
    """Show the home page and prediction form."""
    countries = list_countries()
    return render_template("index.html", countries=countries)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form input, predict HDI, and show the result page."""
    try:
        model = load_model()

        life_expectancy = float(request.form["life_expectancy"])
        mean_schooling = float(request.form["mean_schooling"])
        expected_schooling = float(request.form["expected_schooling"])
        gni_per_capita = float(request.form["gni_per_capita"])
        country_name = request.form.get("country_name") or None

        values = [life_expectancy, mean_schooling, expected_schooling, gni_per_capita]
        validate_inputs(values)

        input_data = pd.DataFrame([values], columns=FEATURE_COLUMNS)

        predicted_score = round(float(model.predict(input_data)[0]), 3)
        predicted_score = max(0, min(1, predicted_score))
        category = classify_hdi(predicted_score)
        prediction_id = save_prediction_record(
            values, predicted_score, category, country_name
        )

        return render_template(
            "result.html",
            score=f"{predicted_score:.3f}",
            category=category,
            message=category_message(category),
            prediction_id=prediction_id,
        )

    except FileNotFoundError as error:
        return render_template("result.html", error=str(error))
    except ValueError as error:
        return render_template("result.html", error=str(error))
    except Exception as error:
        return render_template(
            "result.html",
            error=f"Something went wrong while making the prediction: {error}",
        )


@app.route("/history")
def history():
    """Show recent database-backed HDI predictions."""
    predictions = get_recent_predictions(limit=10)
    return render_template("history.html", predictions=predictions)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)
