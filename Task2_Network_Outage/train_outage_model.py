# train_outage_model.py
# -------------------------------------
# This script processes historical network outage data, trains a RandomForestRegressor 
# to predict outage duration based on features (equipment age, maintenance history, weather conditions, and traffic load),
# and saves the trained model along with the feature columns for later use.

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def process_data(csv_file):
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    # Convert outage_start and outage_end to datetime and compute duration in minutes
    df['outage_start'] = pd.to_datetime(df['outage_start'])
    df['outage_end'] = pd.to_datetime(df['outage_end'])
    df['duration'] = (df['outage_end'] - df['outage_start']).dt.total_seconds() / 60.0

    # Convert equipment_age (e.g., "5 years") to numeric
    df['equipment_age'] = df['equipment_age'].str.extract(r'(\d+)').astype(float)

    # One-hot encode categorical features
    categorical_features = ['maintenance_history', 'weather_conditions', 'traffic_load', 'network_area']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Define features and target (predicting outage duration)
    X = df_encoded.drop(columns=['outage_id', 'outage_start', 'outage_end', 'cause', 'duration'])
    y = df_encoded['duration']
    return X, y

def train_predictive_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse:.2f}")
    
    # Compute and print feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))
    
    return model, list(X.columns)

if __name__ == "__main__":
    csv_file = "network_outages.csv"  # Ensure this file is in the project directory
    print("Processing data...")
    X, y = process_data(csv_file)
    
    print("Training predictive model...")
    model, feature_columns = train_predictive_model(X, y)
    
    # Save the model and feature columns
    model_data = {"model": model, "feature_columns": feature_columns}
    with open("outage_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model training complete. Model saved as 'outage_model.pkl'.")
