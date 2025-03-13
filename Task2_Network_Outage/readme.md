# Network Outage Chatbot and Prediction

This project enhances network operations support by providing an AI-powered chatbot and a predictive machine learning model for forecasting network outages. The solution consists of two parts:

1. **Outage Model Training:** A script that processes historical network outage data (from `network_outages.csv`), trains a RandomForestRegressor to predict outage duration, and saves the trained model.
2. **Interactive UI:** A Streamlit app that fetches outage details (simulating an API call), generates a summary, and predicts outage duration based on user-provided features.

> **Note:** Ensure you have the required CSV file (`network_outages.csv`) in the project directory.

---

## Prerequisites

- **Python 3.8 or higher**
- **Virtual Environment:** It is recommended to use a virtual environment.
- **CSV File:** Place `network_outages.csv` (containing outage details) in the project root.
- **Dependencies:** Install Python packages as listed in `requirements.txt`.

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment

**On macOS/Linux:**
```bash
python -m venv outage_env
source outage_env/bin/activate
```

**On Windows:**
```bash
python -m venv outage_env
outage_env\Scripts\activate
```

### 2. Install Dependencies

Create a `requirements.txt` file with the following content:
```plaintext
pandas
numpy
scikit-learn
streamlit
nest_asyncio
```
Then install:
```bash
pip install -r requirements.txt
```

---

## Running the Project

### A. Training the Predictive Model

If you want to train your own ML model using your CSV data, run:
```bash
python train_outage_model.py
```
- This script will process the CSV data, compute outage duration, train a RandomForestRegressor, and save the model along with the feature columns in `outage_model.pkl`.

### B. Running the Interactive UI

Once the model is trained (or if you already have `outage_model.pkl` provided), launch the interactive UI with:
```bash
streamlit run app_outage.py
```
- The app will allow you to:
  - **Fetch Outage Summary:** Enter an outage ID (e.g., "001" or "12"—the app will automatically format IDs) to retrieve and display a summary of the outage.
  - **Predict Outage Duration:** Input details such as equipment age, maintenance history, weather conditions, traffic load, and network area to predict the outage duration.

---

## CSV Data Format

Your `network_outages.csv` should have the following columns:
- `outage_id`
- `network_area`
- `outage_start` (format: M/D/YYYY HH:MM, e.g., "3/18/2025 17:00")
- `outage_end` (format: M/D/YYYY HH:MM, e.g., "3/18/2025 20:00")
- `cause`
- `affected_customers`
- `equipment_age` (e.g., "2 years")
- `maintenance_history` (e.g., "Regular")
- `weather_conditions` (e.g., "Windy")
- `traffic_load` (e.g., "Medium")

Example row:
```csv
001,Suburbs,3/18/2025 17:00,3/18/2025 20:00,Equipment failure,441,2 years,Regular,Windy,Medium
```

---

## Troubleshooting

- **Outage ID Formatting:**  
  The app now converts user input to a zero-padded 3-digit string. For example, entering "12" will be interpreted as "012" to match the CSV data.

- **CSV File Not Found:**  
  Ensure that `network_outages.csv` is in the same directory as the Python scripts.

- **Model File Missing:**  
  If `outage_model.pkl` is not found, run `train_outage_model.py` first.

---

## Summary

This project transitions our network operations support system to an AI-powered solution by integrating a RAG-based chatbot for outage summaries and a predictive ML model for outage forecasting. Follow the above steps to train the model and run the interactive UI.