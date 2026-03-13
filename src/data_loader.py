import pandas as pd
import numpy as np
import os

def load_data(data_dir="data"):
    """
    Loads all relevant CSV datasets and returns them.
    """
    accounts_path = os.path.join(data_dir, "accounts.csv")
    customers_path = os.path.join(data_dir, "customers.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")
    fraud_labels_path = os.path.join(data_dir, "fraud_labels.csv")

    try:
        accounts = pd.read_csv(accounts_path)
        customers = pd.read_csv(customers_path)
        transactions = pd.read_csv(transactions_path)
        fraud_labels = pd.read_csv(fraud_labels_path)
        
        print(f"Loaded {len(customers)} customers.")
        print(f"Loaded {len(accounts)} accounts.")
        print(f"Loaded {len(transactions)} transactions.")
        print(f"Loaded {len(fraud_labels)} fraud labels.")
        
        return customers, accounts, transactions, fraud_labels
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None

def merge_data(customers, accounts, transactions, fraud_labels):
    """
    Merges datasets into a single comprehensive dataframe for modeling.
    """
    if customers is None or accounts is None or transactions is None or fraud_labels is None:
        return None
        
    #Merge transactions with fraud labels
    df = pd.merge(transactions, fraud_labels, on='transaction_id', how='left')
    
    #Merge with accounts to get customer_id
    df = pd.merge(df, accounts, on='account_id', how='left')
    
    #Merge with customers to get demographic data
    df = pd.merge(df, customers, on='customer_id', how='left')
    
    print(f"Merged dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    return df

def clean_data(df):
    """
    Perform basic data cleaning on the merged dataframe.
    """
    if df is None:
        return None
        
    if 'fraud_flag' in df.columns:
        # Assuming unlabelled transactions are legitimate for now, or we drop them
        # For simplicity, filling missing flags with 0
        df['fraud_flag'] = df['fraud_flag'].fillna(0)
        
    return df

def get_processed_data(data_dir="data"):
    customers, accounts, transactions, fraud_labels = load_data(data_dir)
    df = merge_data(customers, accounts, transactions, fraud_labels)
    df = clean_data(df)
    return df

if __name__ == "__main__":
    df = get_processed_data()
    if df is not None:
        print(df.head())
        print(df.info())
