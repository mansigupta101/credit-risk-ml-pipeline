"""Translates model predictions into financial impact -- dollar-cost confusion matrix."""

import pandas as pd
import json

PREDICTIONS_PATH = "ml/artifacts/test_predictions.csv"
OUTPUT_PATH = "ml/artifacts/financial_impact.json"

THRESHOLD = 0.5
LOSS_GIVEN_DEFAULT = 0.6   # fraction of loan lost when a default happens
PROFIT_MARGIN = 0.08       # interest margin earned on a performing loan


def classify(df):
    df = df.copy()
    df["rejected"] = df["xgb_prob"] >= THRESHOLD

    # false positive here = approved a loan that defaulted (model said "safe", was wrong)
    df["is_fp"] = (~df["rejected"]) & (df["defaulted"] == 1)
    # false negative = rejected a loan that would have performed fine
    df["is_fn"] = (df["rejected"]) & (df["defaulted"] == 0)
    # true positive = correctly rejected a defaulter
    df["is_tp"] = (df["rejected"]) & (df["defaulted"] == 1)
    # true negative = correctly approved a good customer
    df["is_tn"] = (~df["rejected"]) & (df["defaulted"] == 0)

    return df


def compute_financial_impact(df):
    fp_cost = (df.loc[df["is_fp"], "loan_amount_requested"] * LOSS_GIVEN_DEFAULT).sum()
    fn_cost = (df.loc[df["is_fn"], "loan_amount_requested"] * PROFIT_MARGIN).sum()
    tp_avoided_loss = (df.loc[df["is_tp"], "loan_amount_requested"] * LOSS_GIVEN_DEFAULT).sum()
    tn_profit = (df.loc[df["is_tn"], "loan_amount_requested"] * PROFIT_MARGIN).sum()

    # baseline: what bad debt would be if every loan were approved (no model)
    baseline_bad_debt = (df.loc[df["defaulted"] == 1, "loan_amount_requested"] * LOSS_GIVEN_DEFAULT).sum()
    model_bad_debt = fp_cost  # only undetected defaulters cause losses under the model

    bad_debt_reduction_pct = (baseline_bad_debt - model_bad_debt) / baseline_bad_debt * 100

    total_good = (df["defaulted"] == 0).sum()
    retained_good_pct = df.loc[df["is_tn"]].shape[0] / total_good * 100

    return {
        "assumptions": {
            "threshold": THRESHOLD,
            "loss_given_default": LOSS_GIVEN_DEFAULT,
            "profit_margin": PROFIT_MARGIN,
        },
        "dollar_confusion_matrix": {
            "false_positive_cost_defaults_approved": round(fp_cost, 2),
            "false_negative_cost_good_customers_rejected": round(fn_cost, 2),
            "true_positive_loss_avoided": round(tp_avoided_loss, 2),
            "true_negative_profit_earned": round(tn_profit, 2),
        },
        "business_impact": {
            "baseline_bad_debt_no_model": round(baseline_bad_debt, 2),
            "bad_debt_with_model": round(model_bad_debt, 2),
            "bad_debt_reduction_pct": round(bad_debt_reduction_pct, 1),
            "profitable_borrowers_retained_pct": round(retained_good_pct, 1),
        },
    }


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    df = classify(df)
    impact = compute_financial_impact(df)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(impact, f, indent=2)

    print(f"Bad debt reduction: {impact['business_impact']['bad_debt_reduction_pct']}%")
    print(f"Profitable borrowers retained: {impact['business_impact']['profitable_borrowers_retained_pct']}%")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
