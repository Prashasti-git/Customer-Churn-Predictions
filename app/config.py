"""
===============================================================================
CUSTOMER CHURN PREDICTION - CONFIGURATION FILE (config.py)
===============================================================================
This file centralizes all constants, column names, options, and model paths used 
throughout the Streamlit application. Having a single configuration file makes 
the code clean, modular, easy to maintain, and beginner-friendly.
"""

import os

# -----------------------------------------------------------------------------
# 1. FILE & MODEL PATHS
# -----------------------------------------------------------------------------
# Define the path to the trained Logistic Regression model file (.pkl)
# We construct the absolute path relative to the root directory for reliability.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")

# -----------------------------------------------------------------------------
# 2. EXACT FEATURE NAMES AND SEQUENCE IN THE TRAINED MODEL (40 FEATURES)
# -----------------------------------------------------------------------------
# The machine learning model was trained on a specific 40-feature structure in X_train.
# The user input MUST be converted into a DataFrame matching this exact order.
FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "MultipleLines_No",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check"
]

# -----------------------------------------------------------------------------
# 3. CATEGORICAL & BINARY COLUMNS LISTS
# -----------------------------------------------------------------------------
# Binary features mapped directly to 0 and 1
BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# 3-category columns that were one-hot encoded using pd.get_dummies()
THREE_CAT_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract"
]

# Nominal 4-category column that was one-hot encoded
MULTI_CAT_COLUMNS = ["PaymentMethod"]

# -----------------------------------------------------------------------------
# 4. DROPDOWN & INPUT SELECTION OPTIONS FOR THE STREAMLIT UI
# -----------------------------------------------------------------------------
GENDER_OPTIONS = ["Female", "Male"]
SENIOR_CITIZEN_OPTIONS = ["No", "Yes"]
BINARY_YES_NO_OPTIONS = ["No", "Yes"]

MULTIPLE_LINES_OPTIONS = ["No", "Yes", "No phone service"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
ADDITIONAL_SERVICES_OPTIONS = ["No", "Yes", "No internet service"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]

# -----------------------------------------------------------------------------
# 5. SAMPLE PRESET PROFILES FOR EASY TESTING IN THE APP
# -----------------------------------------------------------------------------
SAMPLE_PROFILES = {
    "Select Profile": None,
    "High Risk (Likely Churn)": {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00
    },
    "Low Risk (Likely Stay)": {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.00,
        "TotalCharges": 3900.00
    }
}
