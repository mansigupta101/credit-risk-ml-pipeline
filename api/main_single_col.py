"""FastAPI service for real-time credit risk decisions."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


DEMO_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Credit Risk Demo</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 480px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 20px; }
  label { display: block; margin-top: 10px; font-size: 13px; color: #333; }
  input { width: 100%; padding: 6px; box-sizing: border-box; }
  button { margin-top: 16px; padding: 10px 20px; background: #222; color: white; border: none; border-radius: 4px; cursor: pointer; }
  #result { margin-top: 20px; padding: 14px; border-radius: 6px; font-weight: bold; display: none; }
  .approve { background: #e6f4ea; color: #1e7e34; }
  .reject { background: #fdecea; color: #c62828; }
</style>
</head>
<body>
<h1>Credit Risk Decision Demo</h1>
<p style="color:#666; font-size: 13px;">Fill in an applicant's details and check the model's decision.</p>

<label>Age</label><input id="age" type="number" value="35">
<label>Annual Income</label><input id="annual_income" type="number" value="60000">
<label>Credit Score</label><input id="credit_score" type="number" value="700">
<label>Existing Debt</label><input id="existing_debt" type="number" value="5000">
<label>Loan Amount Requested</label><input id="loan_amount_requested" type="number" value="10000">
<label>Employment Years</label><input id="employment_years" type="number" value="5">
<label>Debt-to-Income Ratio</label><input id="debt_to_income_ratio" type="number" step="0.01" value="0.08">
<label>Utilization (3mo avg %)</label><input id="utilization_3mo_avg" type="number" value="30">
<label>Utilization (6mo avg %)</label><input id="utilization_6mo_avg" type="number" value="28">
<label>Utilization (12mo avg %)</label><input id="utilization_12mo_avg" type="number" value="25">
<label>Age WoE score</label><input id="age_woe" type="number" step="0.001" value="0.02">

<button onclick="checkApplication()">Check Application</button>

<div id="result"></div>

<script>
async function checkApplication() {
  const fields = ["age","annual_income","credit_score","existing_debt","loan_amount_requested",
                   "employment_years","debt_to_income_ratio","utilization_3mo_avg",
                   "utilization_6mo_avg","utilization_12mo_avg","age_woe"];
  const payload = {};
  fields.forEach(f => payload[f] = parseFloat(document.getElementById(f).value));

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();

  const box = document.getElementById("result");
  box.style.display = "block";
  box.className = data.decision === "approve" ? "approve" : "reject";
  box.innerHTML = `Decision: ${data.decision.toUpperCase()}<br>Default probability: ${(data.default_probability * 100).toFixed(1)}%`;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return DEMO_PAGE


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
