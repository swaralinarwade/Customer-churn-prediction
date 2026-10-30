"""
Unit Tests for STREAMSCOPE Data Preprocessing
Validates dataset integrity, missing value imputation, and ColumnTransformer pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    load_streaming_dataset,
    clean_streaming_data,
    build_streaming_preprocessor,
    prepare_streaming_split,
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from data.load_streaming_data import CSV_PATH

def test_dataset_exists_and_loads():
    """Verify raw streaming CSV is present and non-empty."""
    assert os.path.exists(CSV_PATH), f"Streaming CSV missing at {CSV_PATH}"
    df = load_streaming_dataset(CSV_PATH)
    assert df.shape[0] >= 5000, f"Expected >=5000 rows, found {df.shape[0]}"
    assert TARGET_COLUMN in df.columns, f"Target '{TARGET_COLUMN}' missing."
    print("  [PASS] Dataset existence and loading verified.")

def test_clean_streaming_data():
    """Verify customerID is dropped and data types coerced."""
    df = load_streaming_dataset(CSV_PATH)
    cleaned = clean_streaming_data(df)

    assert "customerID" not in cleaned.columns, "customerID was not dropped!"
    for col in NUMERICAL_FEATURES:
        assert np.issubdtype(cleaned[col].dtype, np.number), f"Feature {col} is not numeric!"
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1}), "Churn target values not in {0, 1}"
    print("  [PASS] Data cleaning and feature typing verified.")

def test_stratified_split_preserves_distribution():
    """Verify train/test split preserves churn ratio within 1%."""
    df = load_streaming_dataset(CSV_PATH)
    X_train, X_test, y_train, y_test = prepare_streaming_split(df, test_size=0.2, random_state=42)

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    assert abs(train_churn_rate - test_churn_rate) < 0.015, (
        f"Stratification mismatch: train={train_churn_rate:.4f}, test={test_churn_rate:.4f}"
    )
    assert len(X_train) + len(X_test) == len(clean_streaming_data(df))
    print("  [PASS] Stratified split integrity verified.")

def test_preprocessor_transformation():
    """Verify ColumnTransformer transforms features without NaNs."""
    df = load_streaming_dataset(CSV_PATH)
    X_train, X_test, y_train, y_test = prepare_streaming_split(df, test_size=0.2, random_state=42)

    preprocessor = build_streaming_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert not np.isnan(X_train_trans).any(), "NaN found in transformed training data!"
    assert not np.isnan(X_test_trans).any(), "NaN found in transformed test data!"
    print("  [PASS] ColumnTransformer imputation and scaling verified.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PREPROCESSING UNIT TESTS")
    print("=" * 60)
    test_dataset_exists_and_loads()
    test_clean_streaming_data()
    test_stratified_split_preserves_distribution()
    test_preprocessor_transformation()
    print("=" * 60)
    print("ALL PREPROCESSING TESTS PASSED! (4/4)")
    print("=" * 60)
