"""
Automated Test Suite for Customer Churn Prediction Pipeline
Validates data loading, preprocessing, model loading, schema compatibility, and predictions.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import (
    load_raw_dataset,
    clean_dataset,
    prepare_train_test_split,
    build_column_preprocessor,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_customer_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
ALL_MODELS_PATH = os.path.join(BASE_DIR, "models", "all_models.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

def test_dataset_exists_and_valid():
    """Verify raw dataset is present with valid shape and columns."""
    assert os.path.exists(DATA_PATH), f"Dataset file missing at {DATA_PATH}"
    df = load_raw_dataset(DATA_PATH)
    assert df.shape[0] > 1000, f"Expected >1000 rows, found {df.shape[0]}"
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' missing."
    print("  [PASS] Dataset existence and structure verified.")

def test_preprocessing_and_cleaning():
    """Verify handling of blank spaces in TotalCharges and customerID removal."""
    raw_df = load_raw_dataset(DATA_PATH)
    cleaned = clean_dataset(raw_df)
    
    assert "customerID" not in cleaned.columns, "customerID was not dropped!"
    assert cleaned["TotalCharges"].isnull().sum() == 0, "NaNs remain in TotalCharges!"
    assert np.issubdtype(cleaned["TotalCharges"].dtype, np.number), "TotalCharges is not numeric!"
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1}), "Churn target is not encoded to 0 and 1!"
    print("  [PASS] Data cleaning and type conversion verified.")

def test_stratified_split():
    """Verify stratified split maintains churn ratio."""
    raw_df = load_raw_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_train_test_split(raw_df, test_size=0.2, random_state=42)
    
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.01, f"Stratification failed: train={train_ratio}, test={test_ratio}"
    assert len(X_train) + len(X_test) == len(raw_df), "Split size does not match total rows."
    print("  [PASS] Stratified train/test split verified.")

def test_model_loading_and_inference():
    """Verify saved model pipeline loads and makes predictions on raw feature input."""
    assert os.path.exists(MODEL_PATH), f"Saved model missing at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)

    # Synthetic test sample matching schema
    test_input = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 75.0
    }])

    # Verify all expected columns exist
    for col in ALL_FEATURE_COLUMNS:
        assert col in test_input.columns, f"Input missing column: {col}"

    pred = model.predict(test_input)
    proba = model.predict_proba(test_input)

    assert pred[0] in [0, 1], f"Invalid prediction: {pred[0]}"
    assert 0.0 <= proba[0][1] <= 1.0, f"Probability out of bounds: {proba[0][1]}"
    print("  [PASS] Model loading and inference pipeline verified.")

def test_all_models_and_metadata():
    """Verify all 3 candidate models and metadata file."""
    assert os.path.exists(ALL_MODELS_PATH), "all_models.joblib missing!"
    all_models = joblib.load(ALL_MODELS_PATH)
    for expected in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        assert expected in all_models, f"Model {expected} missing from all_models.joblib"

    assert os.path.exists(METADATA_PATH), "model_metadata.json missing!"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "best_model" in meta
    assert "model_evaluations" in meta
    assert len(meta["model_evaluations"]) == 3
    print("  [PASS] All 3 candidate models and metadata verified.")

def run_all_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED PIPELINE TEST SUITE")
    print("=" * 60)
    test_dataset_exists_and_valid()
    test_preprocessing_and_cleaning()
    test_stratified_split()
    test_model_loading_and_inference()
    test_all_models_and_metadata()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
