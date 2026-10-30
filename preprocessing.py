"""
Data Preprocessing Pipeline for STREAMSCOPE
Handles streaming subscriber data ingestion, cleaning, imputation,
encoding, scaling, and stratified train/test partitioning without data leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Canonical Streaming Subscriber Feature Definitions
NUMERICAL_FEATURES = [
    "MonthlyCharges",
    "TenureMonths",
    "AverageWeeklyHours",
    "DaysSinceLastActive",
    "NumberOfProfiles",
    "ContentDownloadsPerMonth",
    "CustomerSupportTickets"
]

CATEGORICAL_FEATURES = [
    "SubscriptionPlan",
    "PrimaryDevice",
    "PaymentMethod",
    "FavoriteGenre"
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Churn"

def load_streaming_dataset(filepath: str) -> pd.DataFrame:
    """Load streaming dataset and perform basic validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    return pd.read_csv(filepath)

def clean_streaming_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans subscriber dataset:
    - Removes non-predictive customerID
    - Validates data types
    - Checks for duplicate records
    """
    cleaned = df.copy()

    # Drop identifier
    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])

    # Drop duplicates if any
    cleaned = cleaned.drop_duplicates()

    # Ensure numerical types
    for col in NUMERICAL_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Ensure target is integer
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    return cleaned

def build_streaming_preprocessor() -> ColumnTransformer:
    """
    Constructs Scikit-learn ColumnTransformer:
    - Numerical: Median Imputer -> StandardScaler
    - Categorical: Most Frequent Imputer -> OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_streaming_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Prepares stratified training and testing splits.
    Preserves true churn ratio across both sets.
    """
    cleaned = clean_streaming_data(df)
    
    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' missing from dataframe.")

    X = cleaned[ALL_FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def get_encoded_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract human-readable feature names post-encoding."""
    feature_names = []
    # Add numerical features in order
    feature_names.extend(NUMERICAL_FEATURES)

    # Add one-hot encoded categorical feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(cat_names))

    return feature_names
