"""Train the Human Development Index prediction model.

This script reads dataset.csv, cleans the data, trains a Linear Regression
model, evaluates it, saves the trained model, and creates EDA plots.
"""

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "hdi_model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI per Capita",
]
TARGET_COLUMN = "HDI Score"


def find_column(dataframe, possible_names):
    """Return the first matching column name from a list of options."""
    for column_name in possible_names:
        if column_name in dataframe.columns:
            return column_name
    raise KeyError(
        "Could not find any of these required columns: "
        + ", ".join(possible_names)
    )


def load_and_prepare_data():
    """Load the CSV file and prepare the columns needed for model training."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "dataset.csv was not found. Place the HDI dataset in the project "
            "folder and run this script again."
        )

    print("Loading dataset.csv...")
    df = pd.read_csv(DATASET_PATH)

    # The provided dataset is a wide yearly dataset. These options let the
    # script work with either the raw full file or a simplified CSV.
    hdi_col = find_column(
        df,
        [
            "HDI Score",
            "Human Development Index (2021)",
            "Human Development Index (2020)",
        ],
    )
    life_col = find_column(
        df,
        [
            "Life Expectancy",
            "Life Expectancy at Birth (2021)",
            "Life Expectancy at Birth (2020)",
        ],
    )
    mean_school_col = find_column(
        df,
        [
            "Mean Years of Schooling",
            "Mean Years of Schooling (2021)",
            "Mean Years of Schooling (2020)",
        ],
    )
    expected_school_col = find_column(
        df,
        [
            "Expected Years of Schooling",
            "Expected Years of Schooling (2021)",
            "Expected Years of Schooling (2020)",
        ],
    )
    gni_col = find_column(
        df,
        [
            "GNI per Capita",
            "Gross National Income Per Capita (2021)",
            "Gross National Income Per Capita (2020)",
        ],
    )

    # Keep only the columns used by the model and rename them to simple names.
    model_df = df[[life_col, mean_school_col, expected_school_col, gni_col, hdi_col]].copy()
    model_df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    # Convert all selected columns to numeric values. Invalid text becomes NaN.
    for column in model_df.columns:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    # Fill numeric missing values using the mean method.
    model_df = model_df.fillna(model_df.mean(numeric_only=True))

    # Drop duplicate rows if any exist after selecting the model columns.
    model_df = model_df.drop_duplicates()

    print(f"Prepared {len(model_df)} rows for training.")
    return model_df


def create_eda_plots(model_df):
    """Create and save beginner-friendly EDA visualizations."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    correlation = model_df.corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("HDI Feature Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    correlation[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values().plot(kind="barh")
    plt.title("Correlation Matrix: Features vs HDI Score")
    plt.xlabel("Correlation with HDI Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_matrix.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(model_df[TARGET_COLUMN], kde=True, bins=20)
    plt.title("Distribution Plot of HDI Scores")
    plt.xlabel("HDI Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "distribution_plot.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=model_df,
        x="GNI per Capita",
        y=TARGET_COLUMN,
        hue="Life Expectancy",
        palette="viridis",
    )
    plt.title("Scatter Plot: GNI per Capita vs HDI Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scatter_plot.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    long_df = model_df.melt(
        value_vars=FEATURE_COLUMNS,
        var_name="Indicator",
        value_name="Value",
    )
    sns.stripplot(data=long_df, x="Indicator", y="Value", jitter=True, alpha=0.6)
    plt.title("Strip Plot of HDI Input Indicators")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "strip_plot.png"), dpi=200)
    plt.close()

    print(f"EDA plots saved in {PLOTS_DIR}")


def train_model(model_df):
    """Split the data, train the model, evaluate it, and save it."""
    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\nModel Evaluation")
    print("----------------")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"\nTrained model saved as {MODEL_PATH}")


def main():
    """Run the full training workflow."""
    try:
        model_df = load_and_prepare_data()
        create_eda_plots(model_df)
        train_model(model_df)
    except FileNotFoundError as error:
        print(f"Dataset Error: {error}")
    except KeyError as error:
        print(f"Column Error: {error}")
    except Exception as error:
        print(f"Unexpected Error: {error}")


if __name__ == "__main__":
    main()
