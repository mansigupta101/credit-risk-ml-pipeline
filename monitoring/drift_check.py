"""Simulates a macroeconomic shift and checks for data drift against the training baseline."""

import duckdb
import pandas as pd
import numpy as np
import json

from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift

DB_PATH = "include/dbt_project/dev.duckdb"
OUTPUT_PATH = "monitoring/drift_report.json"

DRIFT_COLUMNS = ["debt_to_income_ratio", "annual_income", "utilization_3mo_avg", "credit_score"]
PSI_ALERT_THRESHOLD = 0.2  # standard industry cutoff: >0.2 = significant shift


def load_baseline():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("select * from mart_credit_features").df()
    con.close()
    return df


def simulate_macro_shock(df):
    """
    Simulates a rise in unemployment and inflation:
    incomes drop, debt burdens and utilization rise.
    """
    shocked = df.copy()
    shocked["annual_income"] = shocked["annual_income"] * 0.85
    shocked["debt_to_income_ratio"] = shocked["debt_to_income_ratio"] * 1.4
    shocked["utilization_3mo_avg"] = (shocked["utilization_3mo_avg"] * 1.3).clip(0, 100)
    return shocked


def run_drift_check(reference_df, current_df):
    ref_ds = Dataset.from_pandas(reference_df, data_definition=DataDefinition())
    cur_ds = Dataset.from_pandas(current_df, data_definition=DataDefinition())

    metrics = [ValueDrift(column=col, method="psi", threshold=PSI_ALERT_THRESHOLD) for col in DRIFT_COLUMNS]
    report = Report(metrics)
    snapshot = report.run(cur_ds, ref_ds)

    results = {}
    for m in snapshot.dict()["metrics"]:
        col = m["config"]["column"]
        psi = m["value"]
        results[col] = {
            "psi": round(float(psi), 4),
            "breached": bool(psi > PSI_ALERT_THRESHOLD),
        }
    return results


def main():
    baseline = load_baseline()
    shocked = simulate_macro_shock(baseline)

    results = run_drift_check(baseline, shocked)

    any_breach = any(r["breached"] for r in results.values())

    output = {
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "columns": results,
        "retrain_recommended": any_breach,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    for col, r in results.items():
        flag = "ALERT" if r["breached"] else "ok"
        print(f"{col}: PSI={r['psi']} [{flag}]")

    if any_breach:
        print("Drift detected -- retraining recommended.")
    else:
        print("No significant drift detected.")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
