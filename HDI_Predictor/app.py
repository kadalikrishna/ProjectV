"""Flask app for predicting Human Development Index scores."""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hdi_model.pkl")
FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI per Capita",
]


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


@app.route("/")
def home():
    """Show the home page and prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form input, predict HDI, and show the result page."""
    try:
        model = load_model()

        life_expectancy = float(request.form["life_expectancy"])
        mean_schooling = float(request.form["mean_schooling"])
        expected_schooling = float(request.form["expected_schooling"])
        gni_per_capita = float(request.form["gni_per_capita"])

        input_data = pd.DataFrame(
            [[life_expectancy, mean_schooling, expected_schooling, gni_per_capita]],
            columns=FEATURE_COLUMNS,
        )

        predicted_score = round(float(model.predict(input_data)[0]), 3)
        predicted_score = max(0, min(1, predicted_score))
        category = classify_hdi(predicted_score)

        return render_template(
            "result.html",
            score=f"{predicted_score:.3f}",
            category=category,
            message=category_message(category),
        )

    except FileNotFoundError as error:
        return render_template("result.html", error=str(error))
    except ValueError:
        return render_template(
            "result.html",
            error="Please enter valid numeric values in every field.",
        )
    except Exception as error:
        return render_template(
            "result.html",
            error=f"Something went wrong while making the prediction: {error}",
        )


if __name__ == "__main__":
    app.run(debug=True)
