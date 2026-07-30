"""
===============================================================================
CUSTOMER CHURN PREDICTION - UTILITY FUNCTIONS (utils.py)
===============================================================================
This file handles data preprocessing, validation, model loading, and inference.

CRITICAL PIPELINE REQUIREMENTS:
1. Replicate exact preprocessing used in notebook:
   - Map binary text columns: 'Yes' -> 1, 'No' -> 0
   - Map 'gender': 'Female' -> 0, 'Male' -> 1 (Alphabetical mapping)
   - One-hot encode 3-category columns & PaymentMethod with pd.get_dummies()
   - Reindex columns to match the 40 features of X_train in exact order.
   - Fill missing one-hot columns with 0.
2. Do NOT apply scaling (as no scaler was used in notebook training).
3. Validate numeric inputs to prevent negative values.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from config import MODEL_PATH, FEATURE_NAMES, THREE_CAT_COLUMNS, MULTI_CAT_COLUMNS


@st.cache_resource
def load_model():
    """
    Loads the trained Logistic Regression model from disk using Joblib.
    We use Streamlit's @st.cache_resource decorator to cache the loaded model.
    This prevents reloading the model file on every user interaction, making
    the app fast and efficient.
    
    Returns:
        model: Trained scikit-learn LogisticRegression object.
    """
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
        return None


def validate_inputs(tenure: int, monthly_charges: float, total_charges: float):
    """
    Validates user numeric inputs to ensure they conform to domain rules.
    
    Rules:
    - Tenure cannot be negative.
    - Monthly Charges cannot be negative.
    - Total Charges cannot be negative.
    - Total Charges should logically be consistent with tenure & monthly charges.
    
    Args:
        tenure (int): Customer tenure in months.
        monthly_charges (float): Monthly bill amount.
        total_charges (float): Total bill amount over tenure.
        
    Returns:
        tuple: (is_valid: bool, error_messages: list)
    """
    errors = []
    
    if tenure < 0:
        errors.append("Tenure cannot be negative. Please enter 0 or a positive integer.")
        
    if monthly_charges < 0:
        errors.append("Monthly Charges cannot be negative. Please enter a positive value.")
        
    if total_charges < 0:
        errors.append("Total Charges cannot be negative. Please enter a positive value.")
        
    # Helpful warning for extreme inconsistency (non-blocking, but warning user)
    if tenure > 0 and total_charges == 0 and monthly_charges > 0:
        errors.append("Total Charges is 0 despite positive tenure and monthly charges. Please verify.")
        
    is_valid = len(errors) == 0
    return is_valid, errors


def preprocess_input(raw_input: dict) -> pd.DataFrame:
    """
    Transforms raw user inputs from the Streamlit UI into a 1-row DataFrame 
    that EXACTLY matches the 40-column structure expected by the trained model.
    
    Step-by-step logic matching notebook (data_inspection_and_modification.ipynb):
    1. Create a 1-row pandas DataFrame from user input dictionary.
    2. Convert SeniorCitizen to int (0 or 1).
    3. Binary Map 'Yes' -> 1, 'No' -> 0 for: Partner, Dependents, PhoneService, PaperlessBilling.
    4. Alphabetical Map for 'gender': 'Female' -> 0, 'Male' -> 1.
    5. One-Hot Encode (pd.get_dummies) multi-category columns:
       - 3-category: MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
                     DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract
       - 4-category: PaymentMethod
    6. Reindex against FEATURE_NAMES (40 columns in exact training order).
       Missing columns created during get_dummies are filled with 0.
       
    Args:
        raw_input (dict): Key-value pairs of raw user inputs from Streamlit form.
        
    Returns:
        pd.DataFrame: 1-row DataFrame ready for model prediction.
    """
    # Step 1: Create DataFrame
    df = pd.DataFrame([raw_input])
    
    # Step 2: Ensure numeric types
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').astype(int)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').astype(float)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype(float)
    
    # Step 3: Handle SeniorCitizen ('Yes'/1 -> 1, 'No'/0 -> 0)
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0).astype(int)
        
    # Step 4: Binary Map ('Yes': 1, 'No': 0) for binary columns
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            
    # Step 5: Binary Map for gender (Alphabetical: 'Female' -> 0, 'Male' -> 1)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0).astype(int)
        
    # Step 6: One-Hot Encode 3-category columns & PaymentMethod
    # In notebook: df = pd.get_dummies(df, columns=three_cat_cols, drop_first=False, dtype=int)
    #             df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=False, dtype=int)
    cols_to_dummies = THREE_CAT_COLUMNS + MULTI_CAT_COLUMNS
    df = pd.get_dummies(df, columns=cols_to_dummies, drop_first=False, dtype=int)
    
    # Step 7: Reindex against exact FEATURE_NAMES list (40 features)
    # Any dummy column that was not active in this user input row will be created with value 0.
    # This guarantees exact column alignment and ordering as X_train.
    processed_df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    
    return processed_df


def predict_churn(model, processed_df: pd.DataFrame):
    """
    Makes churn prediction and calculates prediction probabilities using the 
    loaded Logistic Regression model.
    
    Args:
        model: Trained LogisticRegression model instance.
        processed_df (pd.DataFrame): Preprocessed 1-row DataFrame.
        
    Returns:
        dict: Containing prediction class (0 or 1), prediction label, 
              churn probability percentage, stay probability percentage, 
              and confidence score.
    """
    # Generate prediction (0 = Stay, 1 = Churn)
    prediction = int(model.predict(processed_df)[0])
    
    # Generate probabilities for class 0 (Stay) and class 1 (Churn)
    probabilities = model.predict_proba(processed_df)[0]
    prob_stay = float(probabilities[0])
    prob_churn = float(probabilities[1])
    
    # Confidence score is the probability of the predicted class
    confidence = prob_churn if prediction == 1 else prob_stay
    
    return {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "Stay",
        "prob_stay": prob_stay,
        "prob_churn": prob_churn,
        "confidence_percentage": round(confidence * 100, 2),
        "prob_churn_percentage": round(prob_churn * 100, 2),
        "prob_stay_percentage": round(prob_stay * 100, 2)
    }
