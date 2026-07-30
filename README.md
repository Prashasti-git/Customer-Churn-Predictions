# Customer Churn Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict customer churn risk for telecommunication service providers. Powered by a trained **Logistic Regression** model achieving **78.82% accuracy**, this application transforms raw customer profile inputs into actionable retention insights with real-time confidence scores.

---

## Project Overview

Customer churn is one of the most critical metrics for subscription-based businesses. Acquiring new customers can cost up to 5x more than retaining existing ones. This project delivers a complete machine learning lifecycle—from exploratory data analysis and rigorous feature engineering to interactive deployment via a portfolio-quality Streamlit web application.

---

##  Problem Statement

Telecommunication companies face intense competition and customer attrition. The objective is to build a predictive machine learning model and intuitive web application that allows business analysts, customer support representatives, and retention teams to:
1. Identify customers with high risk of canceling their subscriptions (`Churn = 1`).
2. Understand key churn drivers (contract terms, payment methods, internet service tiers).
3. Take proactive retention measures before customer cancellation occurs.

---

## Dataset Description

The project utilizes the **Telco Customer Churn** dataset containing **7,032 customer records** (after cleaning missing total charges) and 21 original feature columns:

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer demographic background. |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | Subscription duration, contract length, billing type. |
| **Financials** | `MonthlyCharges`, `TotalCharges` | Current monthly charges and cumulative total charges. |
| **Phone Services** | `PhoneService`, `MultipleLines` | Voice phone connectivity options. |
| **Internet & Add-ons** | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Internet technology tier (DSL/Fiber) and add-on security/entertainment options. |
| **Target Variable** | `Churn` | Target binary flag (`Yes` = 1, `No` = 0). |

---

## Machine Learning Pipeline

```
[ Raw Dataset: Telco_Customer_Churn.csv ]
                   │
                   ▼
[ Data Cleaning & Missing Value Imputation ] ──► (TotalCharges coerced to numeric, 11 null rows dropped)
                   │
                   ▼
[ Binary & One-Hot Encoding ] ───────────────► (Binary mapped to 0/1; pd.get_dummies on multi-cat features -> 40 columns)
                   │
                   ▼
[ Train-Test Split ] ────────────────────────► (80% Train: 5,625 samples | 20% Test: 1,407 samples, random_state=42)
                   │
                   ▼
[ Model Training & Evaluation ] ─────────────► (Logistic Regression, RF, XGBoost, KNN, SVM, Decision Tree)
                   │
                   ▼
[ Model Serialization ] ─────────────────────► (Saved via Joblib: logistic_regression_model.pkl)
                   │
                   ▼
[ Streamlit Deployment ] ────────────────────► (app.py & utils.py for real-time inference)
```

### Data Preprocessing Details:
- **Binary Mapping**: 
  - `gender`: `Female` ➔ `0`, `Male` ➔ `1` (alphabetical order encoding)
  - `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`: `No` ➔ `0`, `Yes` ➔ `1`
  - `SeniorCitizen`: Binary integer (`0` or `1`)
- **One-Hot Encoding**:
  - `pd.get_dummies(..., drop_first=False, dtype=int)` applied to 3-category columns (`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`) and `PaymentMethod`.
- **Feature Scaling**: No feature scaler (`StandardScaler` / `MinMaxScaler`) was used, preserving raw continuous feature scales for direct interpretability.
- **Final Preprocessed Features**: Exactly **40 columns** in a strict deterministic order.

---

## Model Comparison & Evaluation

Six machine learning classification algorithms were trained and evaluated on the test set:

| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **78.82%** | **0.62** | **0.52** | **0.56** |
| **Random Forest** | 78.18% | 0.62 | 0.48 | 0.54 |
| **XGBoost** | 77.54% | 0.59 | 0.50 | 0.54 |
| **K-Nearest Neighbors (KNN)** | 77.11% | 0.59 | 0.45 | 0.51 |
| **Support Vector Machine (SVM)** | 73.42% | 0.00 | 0.00 | 0.00 |
| **Decision Tree** | 71.93% | 0.47 | 0.49 | 0.48 |

---

## Why Logistic Regression was Selected

1. **Top Performance**: Achieved the highest overall accuracy (**78.82%**) and the highest F1-score (**0.56**) for the minority churn class.
2. **Probability Calibration**: Logistic Regression outputs well-calibrated class probabilities via sigmoid activation, enabling accurate confidence scoring (`predict_proba()`).
3. **Interpretability & Speed**: Provides clear feature coefficients for explaining risk drivers to business stakeholders while executing ultra-fast real-time inference.

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app/
│   ├── app.py                 # Main Streamlit user interface & layout
│   ├── utils.py               # Preprocessing pipeline, validation & model inference
│   ├── config.py              # Centralized constants, column lists & preset profiles
│   └── assets/                # Application images and UI styling assets
│
├── models/
│   └── logistic_regression_model.pkl  # Pre-trained scikit-learn model object
│
├── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb
│   └── data_inspection_and_modification.ipynb
│
├── data/
│   ├── Telco_Customer_Churn.csv # Raw dataset
│   ├── X_train.csv             # Preprocessed training feature matrix (5625 x 40)
│   ├── X_test.csv              # Preprocessed testing feature matrix (1407 x 40)
│   ├── y_train.csv             # Training target labels
│   └── y_test.csv              # Testing target labels
│
├── requirements.txt           # Python package requirements
├── README.md                  # Detailed project documentation
└── .gitignore                 # Environment and bytecode exclusion rules
```

---

## 🚀 Installation & How to Run

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app/app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Application Features

- **Interactive Customer Form**: Multi-tab interface categorized into *Demographics & Account*, *Billing & Contract*, and *Subscribed Services*.
- **Quick Preset Profiles**: Load pre-configured sample profiles (*High Risk Churn* vs. *Low Risk Retention*) in one click from the sidebar.
- **Input Validation**: Automatically prevents invalid or negative inputs for numerical fields (`tenure`, `MonthlyCharges`, `TotalCharges`).
- **Prediction Confidence & Probabilities**: Displays prediction outcome alongside calibrated confidence percentages and visual progress gauges.
- **Risk Driver Insights**: Expandable insight panel explaining top contributing risk factors based on customer parameters.

---

## 🛠️ Technologies Used

- **Language**: Python 3.13
- **Frontend / Web Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Logistic Regression)
- **Model Serialization**: Joblib
- **Visualization**: Matplotlib, Seaborn

---

## 🔮 Future Improvements

- [ ] Implement SHAP / LIME model explainability visualizations in the UI.
- [ ] Add batch prediction support via CSV file upload.
- [ ] Support automated email alerts for high-risk customer predictions.
- [ ] Deploy live onto Streamlit Community Cloud / AWS App Runner.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
