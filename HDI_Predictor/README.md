# Human Development Index (HDI) Predictor

## Project Overview

HDI Predictor is a Python, Machine Learning, and Flask web application that estimates a country's Human Development Index score from health, education, and income indicators.

The app accepts:

- Life Expectancy
- Mean Years of Schooling
- Expected Years of Schooling
- GNI per Capita

It returns a predicted HDI score and classifies the result as Very High, High, Medium, or Low Human Development.

## Problem Statement

Human Development Index combines important social and economic indicators into one score. This project uses a Linear Regression model to predict HDI from four measurable inputs so users can understand how these indicators relate to development level.

## Skills Required

- Python programming
- Data cleaning with Pandas
- Numerical work with NumPy
- Data visualization with Matplotlib and Seaborn
- Machine learning with Scikit-learn
- Flask web development
- HTML and CSS
- Pickle model saving and loading

## Dataset Columns

The model uses these columns after preparing the dataset:

- Life Expectancy
- Mean Years of Schooling
- Expected Years of Schooling
- GNI per Capita
- HDI Score

The included `dataset.csv` is the full HDI dataset. `train_model.py` selects the latest available yearly columns and renames them into the simplified column names above.

## Project Workflow

1. Load `dataset.csv`.
2. Select the HDI, life expectancy, schooling, and income columns.
3. Convert selected columns to numeric values.
4. Fill missing numeric values using the mean method.
5. Generate EDA plots in `static/plots/`.
6. Split data into training and testing sets.
7. Train a Linear Regression model.
8. Evaluate the model with R2 Score, MAE, MSE, and RMSE.
9. Save the trained model as `hdi_model.pkl`.
10. Load the model in Flask and predict HDI from form input.

## Installation Steps

```bash
cd HDI_Predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Train the Model

```bash
python train_model.py
```

This creates:

- `hdi_model.pkl`
- `static/plots/heatmap.png`
- `static/plots/correlation_matrix.png`
- `static/plots/distribution_plot.png`
- `static/plots/scatter_plot.png`
- `static/plots/strip_plot.png`

## How to Initialize the ERD Database

```bash
python init_db.py
```

This creates `hdi_predictor.db` with tables for users, sessions, countries,
datasets, ML models, input data, predictions, and visualization reports.

## How to Run the Flask App

```bash
python app.py
```

Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## Deployment Notes

For platforms such as Render, Railway, or Heroku, set the project root to the
`HDI_Predictor` folder. The included `Procfile` starts the app with Gunicorn:

```text
web: gunicorn app:app
```

The app reads the deployment port from the `PORT` environment variable and keeps
debug mode off unless `FLASK_DEBUG=1` is set.

## Sample Input

- Life Expectancy: `75.5`
- Mean Years of Schooling: `9.8`
- Expected Years of Schooling: `13.2`
- GNI per Capita: `15980`

## Sample Output

```text
Predicted HDI Score: 0.824
HDI Category: Very High Human Development
```

Actual output may vary because it depends on the trained model and dataset values.

## Future Improvements

- Add more models such as Random Forest and Gradient Boosting.
- Compare model performance in a dashboard.
- Add country lookup and historical trend charts.
- Add input validation ranges based on real HDI data.
- Deploy the app online.
