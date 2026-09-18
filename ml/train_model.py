"""Trains and evaluates credit risk models on mart_credit_features."""

import duckdb
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier

DB_PATH = "include/dbt_project/dev.duckdb"
ARTIFACTS_DIR = "ml/artifacts"

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
TARGET = "defaulted"


def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("select * from mart_credit_features").df()
    con.close()
    return df


def split_data(df):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def train_logistic_regression(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, imputer, scaler


def train_xgboost(X_train, y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, imputer=None, scaler=None):
    X_eval = X_test
    if imputer:
        X_eval = imputer.transform(X_eval)
    if scaler:
        X_eval = scaler.transform(X_eval)
    probs = model.predict_proba(X_eval)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "report": classification_report(y_test, preds, output_dict=True),
    }


def generate_shap_plots(xgb_model, X_test):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{ARTIFACTS_DIR}/shap_global_importance.png")
    plt.close()

    # pick a rejected (predicted default) applicant for a local explanation
    probs = xgb_model.predict_proba(X_test)[:, 1]
    rejected_idx = int(np.argmax(probs))

    plt.figure()
    shap.plots.waterfall(shap_values[rejected_idx], show=False)
    plt.tight_layout()
    plt.savefig(f"{ARTIFACTS_DIR}/shap_rejected_applicant.png")
    plt.close()


def main():
    import os
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    lr_model, imputer, scaler = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate(lr_model, X_test, y_test, imputer=imputer, scaler=scaler)

    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate(xgb_model, X_test, y_test)

    print("Logistic Regression -- ROC-AUC:", round(lr_metrics["roc_auc"], 4),
          "PR-AUC:", round(lr_metrics["pr_auc"], 4))
    print("XGBoost             -- ROC-AUC:", round(xgb_metrics["roc_auc"], 4),
          "PR-AUC:", round(xgb_metrics["pr_auc"], 4))

    generate_shap_plots(xgb_model, X_test)

    joblib.dump(xgb_model, f"{ARTIFACTS_DIR}/xgb_model.joblib")
    joblib.dump(lr_model, f"{ARTIFACTS_DIR}/lr_model.joblib")
    joblib.dump(imputer, f"{ARTIFACTS_DIR}/imputer.joblib")
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.joblib")

    metrics = {"logistic_regression": lr_metrics, "xgboost": xgb_metrics}
    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    X_test.assign(defaulted=y_test, xgb_prob=xgb_model.predict_proba(X_test)[:, 1]) \
        .to_csv(f"{ARTIFACTS_DIR}/test_predictions.csv", index=False)

    print(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
