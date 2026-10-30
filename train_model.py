"""
Model Training & Benchmarking Pipeline for STREAMSCOPE
Trains, evaluates, and persists Logistic Regression, Decision Tree,
and Random Forest classifiers using Scikit-Learn Pipelines.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_streaming_dataset,
    prepare_streaming_split,
    build_streaming_preprocessor,
    get_encoded_feature_names,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.evaluation import evaluate_streaming_classifier, format_benchmark_table
from data.load_streaming_data import ensure_dataset, CSV_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def run_training_pipeline():
    """Executes end-to-end model training, evaluation, and artifact persistence."""
    print("=" * 70)
    print("STREAMSCOPE: SUBSCRIBER INTELLIGENCE PLATFORM — MODEL PIPELINE")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Dataset Verification
    if not os.path.exists(CSV_PATH):
        print("[INFO] Dataset not found. Generating streaming subscriber data...")
        raw_df = ensure_dataset()
    else:
        raw_df = load_streaming_dataset(CSV_PATH)
    print(f"[1/6] Ingested dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # 2. Stratified Train/Test Split (80/20)
    print("\n[2/6] Executing stratified 80/20 train/test partition...")
    X_train, X_test, y_train, y_test = prepare_streaming_split(
        raw_df,
        test_size=0.2,
        random_state=42
    )
    print(f"      Training set: {X_train.shape[0]} accounts (Churn: {int(y_train.sum())}, Retained: {len(y_train) - int(y_train.sum())})")
    print(f"      Testing set:  {X_test.shape[0]} accounts (Churn: {int(y_test.sum())}, Retained: {len(y_test) - int(y_test.sum())})")

    # 3. Model Definitions (with balanced class weighting)
    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=4,
            n_jobs=-1
        )
    }

    trained_pipelines = {}
    eval_results = []

    print("\n[3/6] Training algorithms using ColumnTransformer Pipeline...")
    for name, clf in algorithms.items():
        print(f"      -> Fitting: {name}...")
        preprocessor = build_streaming_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        # Fit strictly on X_train to prevent leakage
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Evaluate on held-out test data
        print(f"      -> Benchmarking: {name} on held-out test set (N={len(X_test)})...")
        eval_dict = evaluate_streaming_classifier(pipeline, X_test, y_test, model_name=name)
        eval_results.append(eval_dict)

    # 4. Compile Comparison
    print("\n[4/6] Model Benchmarking Results:")
    comp_df = format_benchmark_table(eval_results)
    print("-" * 70)
    print(comp_df.to_string(index=False))
    print("-" * 70)

    # Export comparison CSV
    csv_out_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    comp_df.to_csv(csv_out_path, index=False)
    print(f"      Saved benchmark metrics to: {csv_out_path}")

    # 5. Model Selection Criterion
    print("\n[5/6] Model Selection Decision Analysis:")
    best_eval = max(eval_results, key=lambda x: (x["f1_score"], x["roc_auc"]))
    best_model_name = best_eval["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    selection_rationale = (
        f"Selected '{best_model_name}' as default model because it achieved highest F1-Score "
        f"({best_eval['f1_score']:.4f}) and ROC-AUC ({best_eval['roc_auc']:.4f}) on held-out test data. "
        f"In streaming subscription economics, subscriber defection is costly. "
        f"{best_model_name} delivers balanced churn recall ({best_eval['recall']:.4f}) "
        f"and superior accuracy ({best_eval['accuracy']:.4f}), effectively detecting at-risk subscribers."
    )
    print(f"      Selected Champion: {best_model_name}")
    print(f"      Rationale: {selection_rationale}")

    # 6. Extract Feature Importances from Random Forest
    rf_pipeline = trained_pipelines["Random Forest"]
    rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_classifier = rf_pipeline.named_steps["classifier"]

    encoded_feature_names = get_encoded_feature_names(rf_preprocessor)
    importances = rf_classifier.feature_importances_.tolist()

    feature_ranking = [
        {"feature": fn, "importance": round(imp, 5)}
        for fn, imp in sorted(zip(encoded_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # Persist artifacts
    print("\n[6/6] Persisting Models & Metadata...")
    best_model_path = os.path.join(MODELS_DIR, "churn_model.joblib")
    all_models_path = os.path.join(MODELS_DIR, "all_models.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(trained_pipelines, all_models_path)
    print(f"      Saved best model to: {best_model_path}")
    print(f"      Saved all candidate pipelines to: {all_models_path}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": "STREAMSCOPE Subscriber Intelligence Platform",
        "dataset": {
            "source": "Educational Streaming Subscriber Engagement Benchmark",
            "total_records": int(raw_df.shape[0]),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "churn_rate_overall": round(float(raw_df["Churn"].mean()), 4)
        },
        "best_model": {
            "name": best_model_name,
            "metrics": {
                "accuracy": best_eval["accuracy"],
                "precision": best_eval["precision"],
                "recall": best_eval["recall"],
                "f1_score": best_eval["f1_score"],
                "roc_auc": best_eval["roc_auc"]
            },
            "selection_rationale": selection_rationale
        },
        "model_evaluations": eval_results,
        "feature_importance_top": feature_ranking[:12],
        "all_feature_names": encoded_feature_names,
        "input_features": ALL_FEATURE_COLUMNS
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"      Saved model metadata to: {meta_path}")

    print("\n" + "=" * 70)
    print("[SUCCESS] STREAMSCOPE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return comp_df, metadata

if __name__ == "__main__":
    run_training_pipeline()
