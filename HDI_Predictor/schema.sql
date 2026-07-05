PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    login_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    logout_time DATETIME,
    status VARCHAR(20) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE TABLE IF NOT EXISTS country (
    country_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_name VARCHAR(255) UNIQUE NOT NULL,
    region VARCHAR(100),
    population INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name VARCHAR(255) UNIQUE NOT NULL,
    source VARCHAR(255),
    total_rows INTEGER,
    total_columns INTEGER,
    uploaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_model (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(255) UNIQUE NOT NULL,
    algorithm_used VARCHAR(100) NOT NULL,
    accuracy_score FLOAT,
    r2_score FLOAT,
    model_file_path VARCHAR(255) NOT NULL,
    trained_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hdi_input_data (
    input_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    country_id INTEGER,
    life_expectancy FLOAT NOT NULL,
    mean_years_schooling FLOAT NOT NULL,
    expected_years_schooling FLOAT NOT NULL,
    gni_per_capita FLOAT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id),
    FOREIGN KEY (country_id) REFERENCES country(country_id)
);

CREATE TABLE IF NOT EXISTS hdi_prediction (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    predicted_hdi_score FLOAT NOT NULL,
    hdi_category VARCHAR(50) NOT NULL,
    prediction_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (input_id) REFERENCES hdi_input_data(input_id),
    FOREIGN KEY (model_id) REFERENCES ml_model(model_id)
);

CREATE TABLE IF NOT EXISTS visualization_report (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    graph_path VARCHAR(255) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES hdi_prediction(prediction_id)
);
