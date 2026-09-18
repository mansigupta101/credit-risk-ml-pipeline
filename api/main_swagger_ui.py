"""FastAPI service for real-time credit risk decisions."""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_PATH = "ml/artifacts/xgb_model.joblib"
THRESHOLD = 0.5

app = FastAPI(title="Credit Risk API")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "age",
    "annual_income",
    "credit_score",
    "existing_debt",
    "loan_amount_requested",
    "employment_years",
    "debt_to_income_ratio",
    "utilization_3mo_avg",
    "utilization_6mo_avg",
    "utilization_12mo_avg",
    "age_woe",
]


class Application(BaseModel):
    age: float
    annual_income: float
    credit_score: float
    existing_debt: float
    loan_amount_requested: float
    employment_years: float
    debt_to_income_ratio: float
    utilization_3mo_avg: float
    utilization_6mo_avg: float
    utilization_12mo_avg: float
    age_woe: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(application: Application):
    df = pd.DataFrame([application.dict()])[FEATURES]
    prob = float(model.predict_proba(df)[:, 1][0])
    decision = "reject" if prob >= THRESHOLD else "approve"

    return {
        "default_probability": round(prob, 4),
        "decision": decision,
        "threshold": THRESHOLD,
    }
