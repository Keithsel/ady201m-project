"""
This script trains a ML model to predict car prices using data from an SQLite database.

It performs the following steps:
1. Loads the data from an SQLite database.
2. Splits the data into training and test sets.
3. Builds a machine learning pipeline with preprocessing and regression.
4. Trains the model using GridSearchCV.
5. Evaluates the model performance on the test set.
6. Saves the trained model and preprocessor as pickle files.

Usage:
    python train_regression.py <database_filepath> <table_name> <pipeline_filepath>

Example:
    python train_regression.py cars.db cars_table car_price_model.pkl preprocessor.pkl
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pickle
import argparse

def load_data(database_filepath, table_name):
    """
    Load data from an SQLite database.

    Args:
        database_filepath (str): Filepath for the SQLite database.
        table_name (str): Name of the table containing the data.

    Returns:
        X (DataFrame): Feature DataFrame.
        y (Series): Target Series.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name, engine)
    X = df.drop(columns=['Name', 'Price'])
    y = df['Price']
    return X, y

def build_preprocessor():
    """
    Build preprocessor for numerical and categorical features.

    Returns:
        ColumnTransformer: Preprocessor for the data.
    """
    numerical_features = ['Year', 'Kilometers_Driven', 'Mileage(kmpl)', 'Engine (CC)', 'Power (bhp)', 'Seats']
    categorical_features = ['Automaker', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return preprocessor

def build_model():
    """
    Build machine learning pipeline with GridSearchCV.

    Returns:
        GridSearchCV: GridSearchCV object with pipeline and parameters.
    """
    preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])

    parameters = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 6],
        'regressor__learning_rate': [0.01, 0.1]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring='neg_mean_squared_error', verbose=2)

    return cv

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics.

    Args:
        model (GridSearchCV): Trained model.
        X_test (DataFrame): Test feature data.
        y_test (Series): Test target data.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

def save_pipeline(pipeline, filepath):
    """
    Save the trained pipeline to a pickle file.

    Args:
        pipeline (Pipeline): Trained pipeline.
        filepath (str): Filepath to save the pipeline.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(pipeline, file)

def main():
    parser = argparse.ArgumentParser(description="Train a car price prediction model and save the pipeline.")
    parser.add_argument('database_filepath', type=str, help='Filepath for the SQLite database')
    parser.add_argument('table_name', type=str, help='Name of the table containing the data')
    parser.add_argument('pipeline_filepath', type=str, help='Filepath for the output pipeline pickle file')
    
    args = parser.parse_args()

    print('Loading data...\n    DATABASE: {}'.format(args.database_filepath))
    X, y = load_data(args.database_filepath, args.table_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, y_test)

    print('Saving pipeline...\n    PIPELINE: {}'.format(args.pipeline_filepath))
    save_pipeline(model, args.pipeline_filepath)

    print('Trained model pipeline saved!')

if __name__ == '__main__':
    main()
