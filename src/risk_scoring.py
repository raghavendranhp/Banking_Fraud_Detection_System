import pandas as pd
import numpy as np

def calculate_customer_risk_score(df):
    """
    Calculate a customer risk score based on transaction behavior, 
    balance trends, and high-risk merchant categories.
    """
    if df is None or df.empty:
        return None
        
    df = df.copy()
    
    #We aggregate at the customer level
    customer_risk = df.groupby('customer_id').agg(
        total_tx_volume=('transaction_amount', 'sum'),
        avg_tx_amount=('transaction_amount', 'mean'),
        max_tx_amount=('transaction_amount', 'max'),
        tx_count=('transaction_id', 'count'),
        # Count high risk merchant transactions (assuming risk score > 0.5 is high risk)
        high_risk_merchants_count=('merchant_risk_score', lambda x: (x > 0.5).sum())
    ).reset_index()
    
   

    
    #volume Risk: higher total volume or max transaction relative to peers increases risk
    volume_max = customer_risk['total_tx_volume'].max()
    if volume_max > 0:
        volume_risk = (customer_risk['total_tx_volume'] / volume_max) * 30
    else:
        volume_risk = 0
        
    #Activity Risk: very high frequency could be suspicious
    count_max = customer_risk['tx_count'].max()
    if count_max > 0:
        activity_risk = (customer_risk['tx_count'] / count_max) * 20
    else:
        activity_risk = 0
        
    #Merchant Risk: transactions at high-risk merchants
    merchant_max = customer_risk['high_risk_merchants_count'].max()
    if merchant_max > 0:
        merchant_risk = (customer_risk['high_risk_merchants_count'] / merchant_max) * 50
    else:
        merchant_risk = 0
        
    #Calculate Total Score
    customer_risk['raw_risk_score'] = volume_risk + activity_risk + merchant_risk
    
    #Categorize Risk Score
    def get_risk_label(score):
        if score > 75: return "High"
        elif score > 40: return "Medium"
        else: return "Low"
        
    customer_risk['risk_category'] = customer_risk['raw_risk_score'].apply(get_risk_label)
    
    return customer_risk

if __name__ == "__main__":
    from data_loader import get_processed_data
    from feature_engineering import engineer_features
    
    df = get_processed_data("data")
    if df is not None:
        df_feat = engineer_features(df)
        
        customer_risk = calculate_customer_risk_score(df_feat)
        
        print("\n--- Customer Risk Scores ---")
        print(customer_risk[['customer_id', 'raw_risk_score', 'risk_category']].head(10))
        print("\nRisk Categories Distribution:")
        print(customer_risk['risk_category'].value_counts())
