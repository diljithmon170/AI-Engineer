import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(title="Fraud Detection API", version="2.0")

# ----------------------------------------------------------------------------
# Paths (relative to project root)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# ----------------------------------------------------------------------------
# Load Model & Scaler
# ----------------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"❌ Model file not found at {MODEL_PATH}")

try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    scaler = None  # scaler optional

# ----------------------------------------------------------------------------
# Request Schema
# ----------------------------------------------------------------------------
class Transaction(BaseModel):
    """Transaction with V1..V28 features + Amount"""
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/")
def root():
    return {"message": "Welcome to Fraud Detection API. Use /predict (POST) to check transactions."}


@app.post("/predict")
def predict(tx: Transaction):
    """Predict fraud probability for a given transaction."""
    try:
        # Convert request into ordered array
        features = [
            tx.V1, tx.V2, tx.V3, tx.V4, tx.V5, tx.V6, tx.V7,
            tx.V8, tx.V9, tx.V10, tx.V11, tx.V12, tx.V13, tx.V14,
            tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20, tx.V21,
            tx.V22, tx.V23, tx.V24, tx.V25, tx.V26, tx.V27, tx.V28,
            tx.Amount
        ]
        x = np.array(features).reshape(1, -1)

        # Scale Amount if scaler exists
        if scaler is not None:
            amount = np.array(x[:, -1]).reshape(-1, 1)
            amount_scaled = scaler.transform(amount)
            x[:, -1] = amount_scaled[:, 0]

        # Predict
        proba = model.predict_proba(x)[0, 1]
        pred = int(proba >= 0.5)

        return {"fraud_probability": float(proba), "is_fraud": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
