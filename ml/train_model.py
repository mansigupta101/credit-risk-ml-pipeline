"""Trains and evaluates credit risk models on mart_credit_features."""

import duckdb
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, brier_score_loss
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

    # no class_weight -- trained on the natural class distribution.
    # Reweighting was tested and dropped: it distorted probability calibration
    # without improving discrimination (see reweighted-baseline branch).
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, imputer, scaler


def train_xgboost(X_train, y_train):
    # no scale_pos_weight -- see note above
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
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
    roc_auc = roc_auc_score(y_test, probs)

    return {
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "pr_auc": average_precision_score(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),
        "report": classification_report(y_test, preds, output_dict=True),
    }, probs


def calibrate_model(model, X_train_transformed, y_train, method):
    """
    Wraps an already-fitted model with probability calibration.
    method: 'sigmoid' (Platt scaling) or 'isotonic' (isotonic regression).
    Fits via internal cross-validation on the training set.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    calibrated.fit(X_train_transformed, y_train)
    return calibrated


def plot_calibration_curve(y_test, probs_before, probs_after, model_name, path):
    frac_before, mean_before = calibration_curve(y_test, probs_before, n_bins=10)
    frac_after, mean_after = calibration_curve(y_test, probs_after, n_bins=10)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(mean_before, frac_before, marker="o", label="Before calibration")
    plt.plot(mean_after, frac_after, marker="o", label="After calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed default rate")
    plt.title(f"Calibration curve -- {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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

    # --- Logistic Regression ---
    lr_model, imputer, scaler = train_logistic_regression(X_train, y_train)
    lr_metrics, lr_probs_before = evaluate(lr_model, X_test, y_test, imputer=imputer, scaler=scaler)

    X_train_transformed = scaler.transform(imputer.transform(X_train))
    lr_calibrated = calibrate_model(lr_model, X_train_transformed, y_train, method="sigmoid")
    lr_metrics_calibrated, lr_probs_after = evaluate(lr_calibrated, X_test, y_test, imputer=imputer, scaler=scaler)

    plot_calibration_curve(y_test, lr_probs_before, lr_probs_after,
                            "Logistic Regression (Platt scaling)",
                            f"{ARTIFACTS_DIR}/calibration_logistic_regression.png")

    # --- XGBoost ---
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics, xgb_probs_before = evaluate(xgb_model, X_test, y_test)

    xgb_calibrated = calibrate_model(xgb_model, X_train, y_train, method="isotonic")
    xgb_metrics_calibrated, xgb_probs_after = evaluate(xgb_calibrated, X_test, y_test)

    plot_calibration_curve(y_test, xgb_probs_before, xgb_probs_after,
                            "XGBoost (isotonic regression)",
                            f"{ARTIFACTS_DIR}/calibration_xgboost.png")

    print("Logistic Regression -- ROC-AUC:", round(lr_metrics["roc_auc"], 4),
          "Gini:", round(lr_metrics["gini"], 4),
          "PR-AUC:", round(lr_metrics["pr_auc"], 4),
          "Brier (before):", round(lr_metrics["brier_score"], 4),
          "Brier (after):", round(lr_metrics_calibrated["brier_score"], 4))
    print("XGBoost             -- ROC-AUC:", round(xgb_metrics["roc_auc"], 4),
          "Gini:", round(xgb_metrics["gini"], 4),
          "PR-AUC:", round(xgb_metrics["pr_auc"], 4),
          "Brier (before):", round(xgb_metrics["brier_score"], 4),
          "Brier (after):", round(xgb_metrics_calibrated["brier_score"], 4))

    generate_shap_plots(xgb_model, X_test)

    joblib.dump(xgb_model, f"{ARTIFACTS_DIR}/xgb_model.joblib")
    joblib.dump(xgb_calibrated, f"{ARTIFACTS_DIR}/xgb_model_calibrated.joblib")
    joblib.dump(lr_model, f"{ARTIFACTS_DIR}/lr_model.joblib")
    joblib.dump(lr_calibrated, f"{ARTIFACTS_DIR}/lr_model_calibrated.joblib")
    joblib.dump(imputer, f"{ARTIFACTS_DIR}/imputer.joblib")
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.joblib")

    metrics = {
        "logistic_regression": lr_metrics,
        "logistic_regression_calibrated": lr_metrics_calibrated,
        "xgboost": xgb_metrics,
        "xgboost_calibrated": xgb_metrics_calibrated,
    }
    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    X_test.assign(defaulted=y_test, xgb_prob=xgb_probs_after) \
        .to_csv(f"{ARTIFACTS_DIR}/test_predictions.csv", index=False)

    print(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
