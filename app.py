import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.append(os.path.abspath('src'))
from risk_scoring import calculate_customer_risk_score

#Configuration
st.set_page_config(
    page_title="Banking Fraud Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Load Models
@st.cache_resource
def load_models():
    models_dir = "models"
    try:
        with open(os.path.join(models_dir, "best_fraud_model.pkl"), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(models_dir, "scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(models_dir, "label_encoders.pkl"), 'rb') as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please run the Jupyter Notebook first.")
        return None, None, None

#Load Data
@st.cache_data
def load_data():
    data_path = "data/featured_data.csv"
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error("Featured data not found. Please run the Jupyter Notebook first.")
        return None

#Sidebar
st.sidebar.title(" Bank-Intel")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Overview", "Fraud Detection Engine", "Customer Risk Profiler"])


#Overview Page
if page == "Overview":
    st.title("Banking Transaction Intelligence")
    st.markdown("Monitor transaction flow, summarize fraud impact, and identify macro patterns.")
    
    df = load_data()
    if df is not None:
        #Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_tx = len(df)
        total_fraud = df['fraud_flag'].sum()
        fraud_rate = (total_fraud / total_tx) * 100 if total_tx > 0 else 0
        total_volume = df['transaction_amount'].sum()
        
        col1.metric("Total Transactions", f"{total_tx:,}")
        col2.metric("Flagged Fraud", int(total_fraud))
        col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        col4.metric("Total Volume Processed", f"${total_volume:,.2f}")
        
        st.markdown("---")
        
        #Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Transaction Amounts by Fraud Status")
            # Limit outliers for visualization
            df_plot = df[df['transaction_amount'] < df['transaction_amount'].quantile(0.99)]
            fig = px.box(df_plot, x="fraud_flag", y="transaction_amount", 
                         color="fraud_flag",
                         labels={"fraud_flag": "Fraudulent?", "transaction_amount": "Amount ($)"},
                         title="Amount Distribution: Safe vs Fraud")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.subheader("High Risk Merchants")
            merchant_fraud = df.groupby('merchant_category')['fraud_flag'].mean().reset_index()
            merchant_fraud = merchant_fraud.sort_values(by="fraud_flag", ascending=False)
            fig2 = px.bar(merchant_fraud, x="merchant_category", y="fraud_flag",
                          labels={"merchant_category": "Merchant Type", "fraud_flag": "Historical Fraud Rate"},
                          title="Fraud Rate by Merchant Category")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Recent Transactions Explorer")
        st.dataframe(df.tail(100), use_container_width=True)


#Fraud Detection Page
elif page == "Fraud Detection Engine":
    st.title("Fraud Detection Engine")
    st.markdown("Run specific transactions through the trained Machine Learning model to determine fraud probability.")
    
    model, scaler, encoders = load_models()
    df = load_data()
    
    if model and df is not None:
        st.subheader("Live Transaction Inference")
        
        sample_tx = st.selectbox("Select a Sample Transaction ID to investigate:", df['transaction_id'].tail(50))
        tx_data = df[df['transaction_id'] == sample_tx].iloc[0]
        
        st.write("### Transaction Details")
        st.json({
            "Account ID": tx_data['account_id'],
            "Amount": f"${tx_data['transaction_amount']:.2f}",
            "Merchant": tx_data['merchant_category'],
            "City": tx_data['city'],
            "Location": tx_data['location'],
            "Actual Fraud Label": "FRAUD" if tx_data['fraud_flag'] == 1 else "SAFE"
        })
        
        if st.button("Run ML Model Prediction"):
            #Preprocess this single row
            tx_df = pd.DataFrame([tx_data])
            
            #Drop cols not meant for modeling
            drop_cols = ['transaction_id', 'account_id', 'customer_id', 'timestamp', 'fraud_flag']
            X_infer = tx_df.drop(columns=[c for c in drop_cols if c in tx_df.columns])
            
            #Encode
            for col, le in encoders.items():
                if col in X_infer.columns:
                    X_infer[col] = le.transform(X_infer[col].astype(str))
            
            #Scale
            Numerical_Cols = [c for c in X_infer.columns if c not in encoders.keys()]
            X_infer[Numerical_Cols] = scaler.transform(X_infer[Numerical_Cols])
            
            #Predict
            pred = model.predict(X_infer)[0]
            prob = model.predict_proba(X_infer)[0][1] if hasattr(model, 'predict_proba') else None
            
            st.markdown("---")
            st.subheader("Model Result")
            if pred == 1:
                st.error(f" **ALERT: Transaction Flagged as Fraudulent!**")
            else:
                st.success(f" **Transaction appears Legitimate.**")
                
            if prob is not None:
                st.write(f"Model Confidence (Fraud Probability): **{prob*100:.2f}%**")
                
                # Gauge chart for probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prob > 0.5 else "darkgreen"},
                        'steps' : [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 50], 'color': "lightyellow"},
                            {'range': [50, 100], 'color': "salmon"}],
                        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }))
                st.plotly_chart(fig)

#Customer Risk Profiler
elif page == "Customer Risk Profiler":
    st.title("Customer Risk Profiler")
    st.markdown("Aggregate customer behavior across multiple transactions to identify high-risk accounts.")
    
    df = load_data()
    if df is not None:
        risk_df = calculate_customer_risk_score(df)
        
        #Risk Distribution Summary
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Risk Distribution")
            risk_counts = risk_df['risk_category'].value_counts().reset_index()
            fig = px.pie(risk_counts, values='count', names='risk_category', 
                         color='risk_category', 
                         color_discrete_map={'High':'red', 'Medium':'orange', 'Low':'green'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Top High-Risk Accounts")
            high_risk = risk_df[risk_df['risk_category'] == 'High'].sort_values(by='raw_risk_score', ascending=False)
            if not high_risk.empty:
                 st.dataframe(high_risk[['customer_id', 'raw_risk_score', 'total_tx_volume', 'high_risk_merchants_count']], use_container_width=True)
            else:
                 st.success("No high risk accounts detected.")

        st.markdown("---")
        st.subheader("Deep Dive: Single Customer")
        selected_cust = st.selectbox("Select Customer ID", risk_df['customer_id'].unique())
        cust_profile = risk_df[risk_df['customer_id'] == selected_cust].iloc[0]
        
        st.write(f"**Overall Risk Segment**: {cust_profile['risk_category']} (Score: {cust_profile['raw_risk_score']:.1f}/100)")
        st.write(f"**Total Volume Transacted**: ${cust_profile['total_tx_volume']:,.2f}")
        st.write(f"**Total Transactions**: {cust_profile['tx_count']}")
        st.write(f"**Transactions at High-Risk Merchants**: {cust_profile['high_risk_merchants_count']}")

