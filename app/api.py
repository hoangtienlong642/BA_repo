import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from scipy.stats import ks_2samp

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import SELECTED_FEATURES, BASE_DIR
from app.features import extract_features, extract_features_single
from app import db

app = FastAPI(
    title="Fraud Detection & Real-Time Scoring API",
    description="API for real-time transaction scoring, queue review management, and model monitoring.",
    version="1.0.0"
)

MODEL_PATH = BASE_DIR / "model" / "model.joblib"
_cached_model = None


def load_scoring_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if MODEL_PATH.exists():
        try:
            _cached_model = joblib.load(MODEL_PATH)
            return _cached_model
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    return None


class TransactionSchema(BaseModel):
    step: int = Field(default=1, description="Hour of simulation (1-744)")
    type: str = Field(default="TRANSFER", description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, etc.")
    amount: float = Field(default=1000.0, description="Transaction amount")
    nameOrig: str = Field(default="C12345678", description="Origin customer ID")
    oldbalanceOrg: float = Field(default=1000.0, description="Origin balance before transaction")
    newbalanceOrig: float = Field(default=0.0, description="Origin balance after transaction")
    nameDest: str = Field(default="C87654321", description="Destination customer ID")
    oldbalanceDest: float = Field(default=0.0, description="Destination balance before transaction")
    newbalanceDest: float = Field(default=0.0, description="Destination balance after transaction")


class ReviewSchema(BaseModel):
    action: str = Field(description="Action: APPROVED, DECLINED, ESCALATED")
    note: Optional[str] = Field(default="", description="Analyst review notes")


class BatchPushSchema(BaseModel):
    transactions: List[TransactionSchema]


@app.on_event("startup")
def startup_event():
    db.init_db()
    load_scoring_model()


@app.get("/health")
def health_check():
    model = load_scoring_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


def compute_anomaly_highlights(tx: dict) -> List[str]:
    highlights = []
    amount = float(tx.get("amount", 0.0))
    old_orig = float(tx.get("oldbalanceOrg", 0.0))
    new_orig = float(tx.get("newbalanceOrig", 0.0))
    tx_type = tx.get("type", "")

    if tx_type in ["TRANSFER", "CASH_OUT"]:
        highlights.append(f"High-risk transaction type: {tx_type}")

    if np.isclose(amount, old_orig, atol=1.0) and old_orig > 0:
        highlights.append("Critical Anomaly: Amount equals 100% of origin balance")

    if new_orig == 0 and old_orig > 0:
        highlights.append("Origin account completely zeroed out")

    if tx_type == "TRANSFER" and float(tx.get("newbalanceDest", 0.0)) == 0 and float(tx.get("oldbalanceDest", 0.0)) == 0:
        highlights.append("Destination balance error: zero balance after TRANSFER")

    if amount > 200000:
        highlights.append(f"Large amount transaction (${amount:,.2f} > $200k)")

    return highlights


@app.post("/predict")
def predict_transaction(tx: TransactionSchema):
    tx_dict = tx.dict()
    X_feat = extract_features_single(tx_dict)
    model = load_scoring_model()

    if model is not None and hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_feat)[0, 1])
    else:
        # Heuristic fallback if model not trained yet
        is_transfer_cashout = tx.type in ["TRANSFER", "CASH_OUT"]
        is_zero_dest = tx.newbalanceDest == 0 and tx.oldbalanceDest == 0
        is_empty_orig = tx.newbalanceOrig == 0 and tx.oldbalanceOrg > 0
        if is_transfer_cashout and (is_zero_dest or is_empty_orig or tx.amount > 200000):
            proba = 0.88
        else:
            proba = 0.02

    threshold = 0.05  # Optimal business threshold
    is_fraud = 1 if proba >= threshold else 0

    risk_level = "CRITICAL" if proba >= 0.7 else ("HIGH" if proba >= 0.2 else ("MEDIUM" if proba >= 0.05 else "LOW"))

    tx_id = db.insert_transaction(tx_dict, proba, is_fraud)
    highlights = compute_anomaly_highlights(tx_dict)

    return {
        "transaction_id": tx_id,
        "fraud_probability": round(proba, 4),
        "is_fraud_predicted": is_fraud,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "anomaly_highlights": highlights,
        "features": X_feat.to_dict(orient="records")[0]
    }


@app.post("/push-data")
def push_data(payload: BatchPushSchema):
    results = []
    for tx in payload.transactions:
        res = predict_transaction(tx)
        results.append(res)
    return {"pushed_count": len(results), "items": results}


@app.get("/queue")
def get_review_queue(limit: int = Query(default=50, ge=1, le=200)):
    return db.get_pending_queue(limit=limit)


@app.post("/queue/{transaction_id}/review")
def review_transaction_id(transaction_id: str, payload: ReviewSchema):
    if payload.action not in ["APPROVED", "DECLINED", "ESCALATED"]:
        raise HTTPException(status_code=400, detail="Action must be APPROVED, DECLINED, or ESCALATED")
    db.review_transaction(transaction_id, payload.action, payload.note or "")
    return {"status": "success", "transaction_id": transaction_id, "action": payload.action}


@app.get("/monitoring/metrics")
def get_monitoring_metrics():
    df_reviews = db.get_reviewed_transactions()
    if df_reviews.empty:
        return {
            "total_reviews": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        }

    y_true = df_reviews["ground_truth"].to_numpy()
    y_pred = df_reviews["is_fraud_predicted"].to_numpy()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "total_reviews": len(df_reviews),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }


@app.get("/monitoring/drift")
def get_data_drift():
    df_all = db.get_all_transactions_df()
    if df_all.empty or len(df_all) < 5:
        return {"status": "INSUFFICIENT_DATA", "sample_size": len(df_all), "drift_detected": False, "features_drift": {}}

    # Extract features for incoming stream
    X_incoming = extract_features(df_all)

    # Synthetic / expected baseline distributions for key features
    baseline_stats = {
        "amount": np.random.exponential(scale=150000, size=500),
        "oldbalanceOrg": np.random.exponential(scale=800000, size=500),
        "orig_balance_change": np.random.normal(loc=-50000, scale=100000, size=500),
        "is_transfer_or_cashout": np.random.binomial(n=1, p=0.43, size=500)
    }

    drift_results = {}
    any_drift = False

    for feat, base_dist in baseline_stats.items():
        if feat in X_incoming.columns:
            inc_data = X_incoming[feat].dropna().values
            if len(inc_data) >= 5:
                stat, p_val = ks_2samp(inc_data, base_dist)
                is_drifted = p_val < 0.05
                if is_drifted:
                    any_drift = True
                drift_results[feat] = {
                    "ks_stat": round(float(stat), 4),
                    "p_value": round(float(p_val), 4),
                    "drift_detected": is_drifted
                }

    return {
        "status": "DRIFT_DETECTED" if any_drift else "STABLE",
        "sample_size": len(df_all),
        "drift_detected": any_drift,
        "features_drift": drift_results
    }


@app.get("/monitoring/triggers")
def get_retraining_triggers():
    metrics = get_monitoring_metrics()
    drift = get_data_drift()

    recall = metrics.get("recall")
    drift_detected = drift.get("drift_detected", False)
    total_reviews = metrics.get("total_reviews", 0)

    recall_trigger = (recall is not None) and (total_reviews >= 10) and (recall < 0.80)
    drift_trigger = drift_detected

    should_retrain = recall_trigger or drift_trigger

    reasons = []
    if recall_trigger:
        reasons.append(f"Rolling Recall ({recall:.2%}) dropped below minimum target (80%).")
    if drift_trigger:
        reasons.append("Significant Data Drift detected in incoming feature distributions.")

    return {
        "should_retrain": should_retrain,
        "overall_status": "TRIGGER_RETRAINING" if should_retrain else "HEALTHY",
        "recall_trigger": recall_trigger,
        "drift_trigger": drift_trigger,
        "triggers_activated": reasons
    }
