"""
Model Evaluation and Benchmarking Module for STREAMSCOPE
Evaluates binary classification metrics with particular focus on churn recall,
confusion matrices, ROC curves, and standardized comparisons.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

def evaluate_streaming_classifier(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Classifier") -> dict:
    """
    Evaluates a trained classifier or pipeline on held-out test data.
    Computes all standard binary classification metrics.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
        except Exception:
            y_proba = None
            roc_auc = None
            roc_data = None
    else:
        y_proba = None
        roc_auc = None
        roc_data = None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"], output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"])

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "roc_data": roc_data
    }

def format_benchmark_table(eval_results: list) -> pd.DataFrame:
    """Formats comparison metrics into a structured presentation DataFrame."""
    rows = []
    for res in eval_results:
        rows.append({
            "Algorithm": res["model_name"],
            "Accuracy": res["accuracy"],
            "Precision (Churn)": res["precision"],
            "Recall (Churn)": res["recall"],
            "F1-Score": res["f1_score"],
            "ROC-AUC": res["roc_auc"]
        })
    return pd.DataFrame(rows)
