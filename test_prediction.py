"""
Unit Tests for STREAMSCOPE Model Inference and Prediction
Validates model loading, schema alignment, probability bounds, and persona predictions.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import ALL_FEATURE_COLUMNS

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
ALL_MODELS_PATH = os.path.join(BASE_DIR, "models", "all_models.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

def test_models_exist_and_load():
    """Verify saved joblib pipelines are readable."""
    assert os.path.exists(MODEL_PATH), f"Model file missing at: {MODEL_PATH}"
    assert os.path.exists(ALL_MODELS_PATH), f"All models file missing at: {ALL_MODELS_PATH}"
    
    pipeline = joblib.load(MODEL_PATH)
    assert hasattr(pipeline, "predict"), "Loaded object lacks predict() method!"
    assert hasattr(pipeline, "predict_proba"), "Loaded object lacks predict_proba() method!"
    
    all_models = joblib.load(ALL_MODELS_PATH)
    assert "Logistic Regression" in all_models
    assert "Decision Tree" in all_models
    assert "Random Forest" in all_models
    print("  [PASS] Saved model artifacts and pipelines verified.")

def test_metadata_structure():
    """Verify model_metadata.json content and metrics."""
    assert os.path.exists(METADATA_PATH), f"Metadata file missing at: {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    assert "best_model" in meta
    assert "model_evaluations" in meta
    assert len(meta["model_evaluations"]) == 3
    for eval_item in meta["model_evaluations"]:
        assert 0.0 <= eval_item["accuracy"] <= 1.0
        assert 0.0 <= eval_item["recall"] <= 1.0
        assert 0.0 <= eval_item["f1_score"] <= 1.0
    print("  [PASS] Model metadata schema and values verified.")

def test_high_risk_persona_prediction():
    """Verify inference on dormant, high-risk streaming profile."""
    model = joblib.load(MODEL_PATH)

    # Dormant viewer with 35 days inactivity, 2 hours/week, 1 profile
    high_risk_profile = pd.DataFrame([{
        "SubscriptionPlan": "Basic",
        "MonthlyCharges": 9.99,
        "TenureMonths": 2,
        "AverageWeeklyHours": 2.5,
        "DaysSinceLastActive": 35,
        "PrimaryDevice": "Mobile",
        "NumberOfProfiles": 1,
        "PaymentMethod": "Digital Wallet",
        "ContentDownloadsPerMonth": 0,
        "FavoriteGenre": "Comedy",
        "CustomerSupportTickets": 3
    }])

    pred = model.predict(high_risk_profile)[0]
    prob = model.predict_proba(high_risk_profile)[0][1]

    assert pred in [0, 1]
    assert 0.0 <= prob <= 1.0
    assert prob > 0.50, f"Expected high churn probability for dormant viewer, got {prob:.2f}"
    print(f"  [PASS] High-risk persona predicted: Churn={pred}, Probability={prob*100:.1f}%.")

def test_loyal_persona_prediction():
    """Verify inference on highly engaged, long-tenure family profile."""
    model = joblib.load(MODEL_PATH)

    # Highly active viewer with 0 days inactive, 28 hrs/week, 4 profiles
    loyal_profile = pd.DataFrame([{
        "SubscriptionPlan": "Premium",
        "MonthlyCharges": 19.99,
        "TenureMonths": 36,
        "AverageWeeklyHours": 28.0,
        "DaysSinceLastActive": 1,
        "PrimaryDevice": "Smart TV",
        "NumberOfProfiles": 4,
        "PaymentMethod": "Credit Card",
        "ContentDownloadsPerMonth": 12,
        "FavoriteGenre": "Sci-Fi / Fantasy",
        "CustomerSupportTickets": 0
    }])

    pred = model.predict(loyal_profile)[0]
    prob = model.predict_proba(loyal_profile)[0][1]

    assert pred in [0, 1]
    assert 0.0 <= prob <= 1.0
    assert prob < 0.40, f"Expected low churn probability for loyal viewer, got {prob:.2f}"
    print(f"  [PASS] Loyal persona predicted: Churn={pred}, Probability={prob*100:.1f}%.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MODEL PREDICTION & INFERENCE TESTS")
    print("=" * 60)
    test_models_exist_and_load()
    test_metadata_structure()
    test_high_risk_persona_prediction()
    test_loyal_persona_prediction()
    print("=" * 60)
    print("ALL PREDICTION TESTS PASSED! (4/4)")
    print("=" * 60)
