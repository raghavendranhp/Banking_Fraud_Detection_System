# Banking Transaction Intelligence & Fraud Detection System 

A comprehensive Machine Learning solution designed to detect potentially fraudulent banking transactions, predict customer credit risk, and generate actionable insights on transaction patterns. This project features a full end-to-end ML pipeline and an interactive Streamlit web dashboard.

## Key Features

1. **Fraud Detection Engine**: Utilizes XGBoost (along with other evaluated models) to accurately classify whether a live transaction is legitimate or fraudulent based on behavioral patterns and historical merchant risk.
2. **Customer Risk Profiler**: Aggregates account-level transaction history to assign a composite risk score (Low, Medium, High) based on transaction volume, activity bursts, and exposure to high-risk merchants.
3. **Interactive Dashboard**: A Streamlit frontend providing macro-level insights, a live inference interface, and individual customer profiling.
4. **End-to-End ML Pipeline**: A unified Jupyter Notebook (`01_EDA_and_Modeling.ipynb`) handling everything from data cleaning and feature engineering to model training and serialization.


## Live Demo

---

##  Project Architecture

```
Banking_Fraud_Detection_System/
│
├── app.py                  # Streamlit Dashboard application
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb # Unified ML Pipeline (EDA, Training, Serialization)
├── src/
│   ├── data_loader.py      # Utilities to ingest and merge the initial CSV datasets
│   └── risk_scoring.py     # Logic for aggregating and calculating Customer Risk Scores
├── models/                 # Serialized ML artifacts (Model, Scaler, Encoders)
├── data/                   # Raw and processed datasets (e.g., featured_data.csv)
└── reports/                # Markdown reports with business insights & observations
```

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/raghavendranhp/Banking_Fraud_Detection_System.git
cd Banking_Fraud_Detection_System
python -m venv venv

# Activate the virtual environment:
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter streamlit plotly
```

### 3. Running the Machine Learning Pipeline
Before launching the dashboard, you must run the data pipeline to engineer features and train the specific models.

You can execute the notebook from the terminal:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_and_Modeling.ipynb
```
*(Alternatively, you can open `jupyter lab` and run the cells interactively.)*

This process will create the `models/` directory with the trained XGBoost model and preprocessors, and output `featured_data.csv` into the `data/` folder.

### 4. Launching the Dashboard 
Once the models are generated, start the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

This will open your default web browser to the Bank-Intel interface where you can explore the data, test live transaction fraud detection, and observe customer risk scores.

---

## Evaluation & Insights
- **Insights Report**: Please refer to `reports/insights.md` for a deeper dive into the highest risk transaction patterns, model selection rationale, and strategic recommendations for production deployment. 
- **Models Evaluated**: Logistic Regression, Random Forest, XGBoost, and Isolation Forest. XGBoost was selected as the optimal supervised model based on the F1 Score.
