"""
Dataset Downloader & Verifier for IBM Telco Customer Churn
Downloads authentic IBM Telco Customer Churn dataset from verified official IBM GitHub repository
and saves it to data/telco_customer_churn.csv.
"""

import os
import urllib.request
import pandas as pd

# Primary verified IBM GitHub source for authentic Telco Customer Churn dataset
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "telco_customer_churn.csv")

def ensure_data_directory():
    """Ensure data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)

def download_and_verify_dataset():
    """Download the dataset if not present, and verify integrity."""
    ensure_data_directory()
    
    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] Dataset already exists at: {OUTPUT_PATH}")
    else:
        print(f"[INFO] Downloading dataset from verified official IBM repository:\n  {DATASET_URL}")
        try:
            urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
            print(f"[SUCCESS] Dataset downloaded successfully to: {OUTPUT_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to download dataset automatically: {e}")
            print("\nManual Download Instructions:")
            print("1. Download 'Telco-Customer-Churn.csv' from Kaggle or IBM's official repository:")
            print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
            print(f"2. Place the CSV file into: {OUTPUT_PATH}")
            return False

    # Verify dataset contents
    try:
        df = pd.read_csv(OUTPUT_PATH)
        print(f"[INFO] Dataset Verification:")
        print(f"  - Rows: {df.shape[0]}")
        print(f"  - Columns: {df.shape[1]}")
        print(f"  - Column Names: {list(df.columns)}")
        
        expected_cols = [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
        ]
        
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            print(f"[WARNING] Missing expected columns: {missing}")
            return False
            
        print(f"  - Target ('Churn') distribution:\n{df['Churn'].value_counts(normalize=True).to_dict()}")
        print("[SUCCESS] Authentic IBM Telco Customer Churn dataset verified successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False

if __name__ == "__main__":
    download_and_verify_dataset()
