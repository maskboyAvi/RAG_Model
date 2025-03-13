# app_outage.py
# -------------------------------------
# This Streamlit app integrates outage summary generation and outage duration prediction.
# It fetches outage details from a CSV file (simulating an API), generates a summary,
# and uses a pre-trained RandomForestRegressor model to predict outage duration based on user input features.

import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
import nest_asyncio
nest_asyncio.apply()

# -----------------------------
# Section 1: Fetch Outage Details from CSV
# -----------------------------
def fetch_outage_details(outage_id, csv_file="network_outages.csv"):
    # Load CSV data and force outage_id to be read as a string
    try:
        df = pd.read_csv(csv_file, dtype={'outage_id': str})
    except FileNotFoundError:
        st.error(f"CSV file '{csv_file}' not found!")
        return None

    # Convert user input to integer and then to a 3-digit zero-padded string.
    # This ensures that an input like "12" becomes "012".
    try:
        user_id = str(int(outage_id)).zfill(3)
    except ValueError:
        user_id = outage_id.strip()

    # Filter for the given outage_id (after stripping any whitespace)
    df_filtered = df[df['outage_id'].str.strip() == user_id]
    if df_filtered.empty:
        st.error(f"No outage data found for Outage ID: {user_id}")
        return None
    outage_details = df_filtered.iloc[0].to_dict()
    return outage_details


def generate_outage_summary(details):
    try:
        start = datetime.strptime(details["outage_start"], "%m/%d/%Y %H:%M")
        end = datetime.strptime(details["outage_end"], "%m/%d/%Y %H:%M")
    except Exception as e:
        return f"Error parsing dates: {str(e)}"
    duration_hours = (end - start).total_seconds() / 3600.0
    summary = (f"The outage in the {details['network_area']} area (ID: {details['outage_id']}) lasted approximately "
               f"{duration_hours:.1f} hours and was primarily caused by {details['cause']}. "
               f"The affected equipment was {details['equipment_age']} old, and despite {details['weather_conditions']} weather, "
               f"the {details['traffic_load']} traffic load contributed to the outage. Approximately {details['affected_customers']} customers were affected.")
    return summary

# -----------------------------
# Section 2: Load Trained ML Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("outage_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_columns"]

# -----------------------------
# Section 3: Prepare Input Data for Prediction
# -----------------------------
def prepare_input_data(feature_columns, user_data):
    df_input = pd.DataFrame([user_data])
    categorical_cols = ['maintenance_history', 'weather_conditions', 'traffic_load', 'network_area']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]
    return df_encoded

# -----------------------------
# Section 4: Streamlit UI for Outage Summary and Prediction
# -----------------------------
def main():
    st.set_page_config(page_title="Network Outage Chatbot and Prediction", layout="wide")
    st.title("Network Outage Chatbot and Prediction")
    
    st.header("Outage Summary")
    outage_id = st.text_input("Enter Outage ID to fetch details:", "001")
    if st.button("Get Outage Summary"):
        details = fetch_outage_details(outage_id)
        if details:
            summary = generate_outage_summary(details)
            st.write(summary)
    
    st.header("Predict Network Outage Duration")
    st.write("Enter network outage details to predict outage duration (in minutes).")
    
    col1, col2 = st.columns(2)
    with col1:
        equipment_age = st.number_input("Equipment Age (years):", min_value=0.0, value=5.0)
        maintenance_history = st.selectbox("Maintenance History:", ["Regular", "Irregular"])
        weather_conditions = st.selectbox("Weather Conditions:", ["Clear", "Rainy", "Cloudy", "Stormy"])
    with col2:
        traffic_load = st.selectbox("Traffic Load:", ["High", "Medium", "Low"])
        network_area = st.selectbox("Network Area:", ["Downtown", "Suburban", "Rural"])
    
    if st.button("Predict Outage Duration"):
        user_data = {
            "equipment_age": equipment_age,
            "maintenance_history": maintenance_history,
            "weather_conditions": weather_conditions,
            "traffic_load": traffic_load,
            "network_area": network_area
        }
        model, feature_columns = load_model()
        X_new = prepare_input_data(feature_columns, user_data)
        predicted_duration = model.predict(X_new)[0]
        st.write(f"Predicted outage duration: {predicted_duration:.1f} minutes. This prediction is based on the provided features.")

if __name__ == "__main__":
    main()
