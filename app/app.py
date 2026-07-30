"""
===============================================================================
CUSTOMER CHURN PREDICTION - STREAMLIT WEB APPLICATION (app.py)
===============================================================================
This is the main user interface for the Customer Churn Prediction application.
It integrates with config.py and utils.py to provide a clean, modern, and 
interactive web UI powered by the trained Logistic Regression model.

Key Features:
- Professional Portfolio UI with glassmorphic cards and dynamic CSS.
- Sidebar with Quick Presets (High Risk vs. Low Risk sample profiles).
- Input Validation for numeric features (tenure, monthly/total charges).
- Exact 40-feature preprocessing alignment with the machine learning model.
- Prediction Confidence percentage and probability breakdown.
"""

import streamlit as st
import pandas as pd

# Import modular configuration and helper utilities
from config import (
    GENDER_OPTIONS,
    SENIOR_CITIZEN_OPTIONS,
    BINARY_YES_NO_OPTIONS,
    MULTIPLE_LINES_OPTIONS,
    INTERNET_SERVICE_OPTIONS,
    ADDITIONAL_SERVICES_OPTIONS,
    CONTRACT_OPTIONS,
    PAYMENT_METHOD_OPTIONS,
    SAMPLE_PROFILES
)
from utils import load_model, validate_inputs, preprocess_input, predict_churn

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
# Configures browser tab title, favicon, and wide layout mode
st.set_page_config(
    page_title="Customer Churn Predictor | ML Portfolio App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (PORTFOLIO QUALITY UI)
# -----------------------------------------------------------------------------
# Custom CSS rules for glassmorphic cards, rounded borders, vibrant badges,
# gradient buttons, and responsive metric displays.
CUSTOM_CSS = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Card Styling */
    .header-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Custom Input Form Container */
    .form-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-primary {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-success {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    /* Prediction Banner Cards */
    .result-card-stay {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(4, 120, 87, 0.5) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
        box-shadow: 0 10px 25px rgba(4, 120, 87, 0.2);
    }
    
    .result-card-churn {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.7) 0%, rgba(190, 18, 60, 0.5) 100%);
        border: 1px solid rgba(251, 113, 133, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
        box-shadow: 0 10px 25px rgba(190, 18, 60, 0.2);
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .text-stay { color: #6ee7b7; }
    .text-churn { color: #fca5a5; }
    
    /* Metric boxes */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Submit Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 24px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #4338ca 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
        transform: translateY(-1px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. HELPER FOR PRESET PROFILES IN SESSION STATE
# -----------------------------------------------------------------------------
def apply_preset(preset_dict):
    """Callback to update session state when user selects a preset profile."""
    if preset_dict:
        for key, val in preset_dict.items():
            st.session_state[key] = val


# Initialize session state defaults if not set
default_vals = SAMPLE_PROFILES["High Risk (Likely Churn)"]
for k, v in default_vals.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -----------------------------------------------------------------------------
# 4. SIDEBAR - MODEL METRICS & PRESET SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/38bdf8/user-group.png", width=70)
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("🤖 Model Info")
    st.markdown("""
    <span class="badge badge-primary">Logistic Regression</span>
    <span class="badge badge-success">78.82% Accuracy</span>
    """, unsafe_allow_html=True)
    st.caption("Selected as the top-performing model across 6 evaluated algorithms (Decision Tree, Random Forest, KNN, SVM, XGBoost).")
    
    st.markdown("---")
    st.subheader("⚡ Quick Profile Presets")
    selected_preset = st.selectbox(
        "Load Sample Profile:",
        options=list(SAMPLE_PROFILES.keys()),
        help="Select a pre-configured customer profile to quickly test churn predictions."
    )
    
    if selected_preset and SAMPLE_PROFILES[selected_preset] is not None:
        if st.button("Apply Profile Data"):
            apply_preset(SAMPLE_PROFILES[selected_preset])
            st.rerun()
            
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.markdown("- **Source**: Telco Customer Churn")
    st.markdown("- **Training Rows**: 5,625 customers")
    st.markdown("- **Processed Features**: 40 variables")
    st.markdown("- **Scaling**: Raw feature values")


# -----------------------------------------------------------------------------
# 5. MAIN HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">🔮 Customer Churn Prediction Engine</div>
    <div class="header-subtitle">
        Predict customer retention risk in real-time using our trained Machine Learning pipeline. 
        Fill in customer account parameters below to evaluate retention probability.
    </div>
</div>
""", unsafe_allow_html=True)


# Load the pre-trained model
model = load_model()

if model is None:
    st.error("⚠️ Model file could not be loaded. Please verify `models/logistic_regression_model.pkl` exists.")
    st.stop()


# -----------------------------------------------------------------------------
# 6. INPUT FORM SECTION
# -----------------------------------------------------------------------------
st.subheader("📋 Customer Details Input Form")

with st.form("churn_input_form"):
    
    # Organize input fields into 3 clean, user-friendly tabs
    tab_demo, tab_billing, tab_services = st.tabs([
        "👤 Demographics & Account",
        "💳 Billing & Contract",
        "🌐 Subscribed Services"
    ])
    
    # ------------------- TAB 1: DEMOGRAPHICS -------------------
    with tab_demo:
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Gender",
                options=GENDER_OPTIONS,
                index=GENDER_OPTIONS.index(st.session_state.get("gender", "Female")),
                key="input_gender",
                help="Customer gender"
            )
            
            senior_citizen = st.selectbox(
                "Senior Citizen Status",
                options=SENIOR_CITIZEN_OPTIONS,
                index=SENIOR_CITIZEN_OPTIONS.index(st.session_state.get("SeniorCitizen", "No")),
                key="input_senior",
                help="Is the customer aged 65 or older?"
            )
            
            partner = st.selectbox(
                "Has Partner",
                options=BINARY_YES_NO_OPTIONS,
                index=BINARY_YES_NO_OPTIONS.index(st.session_state.get("Partner", "No")),
                key="input_partner",
                help="Does the customer have a partner?"
            )
            
        with col2:
            dependents = st.selectbox(
                "Has Dependents",
                options=BINARY_YES_NO_OPTIONS,
                index=BINARY_YES_NO_OPTIONS.index(st.session_state.get("Dependents", "No")),
                key="input_dependents",
                help="Does the customer have financial dependents?"
            )
            
            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=120,
                value=int(st.session_state.get("tenure", 2)),
                step=1,
                key="input_tenure",
                help="Number of months customer has stayed with the company."
            )
            
    # ------------------- TAB 2: BILLING & CONTRACT -------------------
    with tab_billing:
        col1, col2 = st.columns(2)
        
        with col1:
            contract = st.selectbox(
                "Contract Type",
                options=CONTRACT_OPTIONS,
                index=CONTRACT_OPTIONS.index(st.session_state.get("Contract", "Month-to-month")),
                key="input_contract",
                help="Customer subscription contract terms."
            )
            
            paperless = st.selectbox(
                "Paperless Billing",
                options=BINARY_YES_NO_OPTIONS,
                index=BINARY_YES_NO_OPTIONS.index(st.session_state.get("PaperlessBilling", "Yes")),
                key="input_paperless",
                help="Whether customer uses paperless billing."
            )
            
            payment_method = st.selectbox(
                "Payment Method",
                options=PAYMENT_METHOD_OPTIONS,
                index=PAYMENT_METHOD_OPTIONS.index(st.session_state.get("PaymentMethod", "Electronic check")),
                key="input_payment",
                help="Payment method preferred by customer."
            )
            
        with col2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=float(st.session_state.get("MonthlyCharges", 85.50)),
                step=0.5,
                format="%.2f",
                key="input_monthly",
                help="Current monthly bill amount."
            )
            
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("TotalCharges", 171.00)),
                step=1.0,
                format="%.2f",
                key="input_total",
                help="Total cumulative bill amount charged to customer."
            )
            
    # ------------------- TAB 3: SERVICES -------------------
    with tab_services:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                options=BINARY_YES_NO_OPTIONS,
                index=BINARY_YES_NO_OPTIONS.index(st.session_state.get("PhoneService", "Yes")),
                key="input_phone"
            )
            
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=MULTIPLE_LINES_OPTIONS,
                index=MULTIPLE_LINES_OPTIONS.index(st.session_state.get("MultipleLines", "No")),
                key="input_multilines"
            )
            
            internet_service = st.selectbox(
                "Internet Service",
                options=INTERNET_SERVICE_OPTIONS,
                index=INTERNET_SERVICE_OPTIONS.index(st.session_state.get("InternetService", "Fiber optic")),
                key="input_internet"
            )
            
        with col2:
            online_security = st.selectbox(
                "Online Security",
                options=ADDITIONAL_SERVICES_OPTIONS,
                index=ADDITIONAL_SERVICES_OPTIONS.index(st.session_state.get("OnlineSecurity", "No")),
                key="input_security"
            )
            
            online_backup = st.selectbox(
                "Online Backup",
                options=ADDITIONAL_SERVICES_OPTIONS,
                index=ADDITIONAL_SERVICES_OPTIONS.index(st.session_state.get("OnlineBackup", "No")),
                key="input_backup"
            )
            
            device_protection = st.selectbox(
                "Device Protection",
                options=ADDITIONAL_SERVICES_OPTIONS,
                index=ADDITIONAL_SERVICES_OPTIONS.index(st.session_state.get("DeviceProtection", "No")),
                key="input_protection"
            )
            
        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                options=ADDITIONAL_SERVICES_OPTIONS,
                index=ADDITIONAL_SERVICES_OPTIONS.index(st.session_state.get("TechSupport", "No")),
                key="input_tech"
            )
            
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=ADDITIONAL_SERVICES_OPTIONS,
                index=ADDITIONAL_SERVICES_OPTIONS.index(st.session_state.get("StreamingTV", "Yes")),
                key="input_tv"
            )
            
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=ADDITIONAL_SERVICES_OPTIONS,
                index=ADDITIONAL_SERVICES_OPTIONS.index(st.session_state.get("StreamingMovies", "Yes")),
                key="input_movies"
            )
            
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🚀 Predict Customer Retention Risk")


# -----------------------------------------------------------------------------
# 7. PREDICTION & RESULTS DISPLAY SECTION
# -----------------------------------------------------------------------------
if submit_button:
    # Gather raw inputs into dictionary
    raw_inputs = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    # 1. Validate inputs
    is_valid, validation_errors = validate_inputs(tenure, monthly_charges, total_charges)
    
    if not is_valid:
        st.error("❌ Input Validation Error:")
        for err in validation_errors:
            st.warning(f"• {err}")
    else:
        # 2. Preprocess input to exact 40 features
        with st.spinner("Processing features and computing prediction probabilities..."):
            processed_df = preprocess_input(raw_inputs)
            
            # 3. Make prediction
            res = predict_churn(model, processed_df)
            
        st.markdown("---")
        st.subheader("🎯 Prediction Output")
        
        # Display Banner Card
        if res["prediction"] == 1:
            st.markdown(f"""
            <div class="result-card-churn">
                <div class="result-title text-churn">⚠️ Customer is Likely to Churn</div>
                <div>High Risk of Account Cancellation</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-stay">
                <div class="result-title text-stay">✅ Customer is Likely to Stay</div>
                <div>Low Risk of Churn - Strong Account Health</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics Breakdown Grid
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Prediction Confidence</div>
                <div class="metric-val">{res['confidence_percentage']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Churn Probability</div>
                <div class="metric-val" style="color: #fca5a5;">{res['prob_churn_percentage']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Retention Probability</div>
                <div class="metric-val" style="color: #6ee7b7;">{res['prob_stay_percentage']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visual Progress Bar
        st.markdown("**Churn Probability Gauge:**")
        st.progress(res["prob_churn"])
        
        # Risk Analysis Explanation Box
        with st.expander("💡 View Risk Factor Insights & Model Drivers"):
            st.markdown("### Top Factors Influencing Prediction:")
            if res["prediction"] == 1:
                st.write("- **Short Tenure**: Customers with low tenure (< 12 months) exhibit significantly higher churn rates.")
                if contract == "Month-to-month":
                    st.write("- **Month-to-Month Contract**: Flexible short-term contracts remove exit barriers for customers.")
                if internet_service == "Fiber optic":
                    st.write("- **Fiber Optic Service**: Higher monthly cost tier correlates with higher price sensitivity.")
                if payment_method == "Electronic check":
                    st.write("- **Electronic Check Payment**: Historically associated with higher churn risk compared to automated bank transfers.")
            else:
                st.write("- **Long-Term Loyalty**: High tenure strengthens retention stability.")
                if contract != "Month-to-month":
                    st.write("- **Annual/Two-Year Contract**: Long-term commitment provides substantial retention stability.")
                if tech_support == "Yes" or online_security == "Yes":
                    st.write("- **Add-on Services (Security/Tech Support)**: High product engagement decreases churn tendency.")


# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Telco Customer Churn Prediction App | Built with Streamlit, Python & Scikit-Learn<br>
    Model: Logistic Regression | Trained on 5,625 records | 40 Features
</div>
""", unsafe_allow_html=True)
