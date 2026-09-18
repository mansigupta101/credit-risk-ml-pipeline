"""
Generates a synthetic, deliberately messy credit application dataset
that simulates real-world banking data problems:
  - missing values (not at random -- correlated with applicant segment)
  - a structural break in how income is reported (annual vs monthly, unlabeled)
  - inconsistent date formats
  - outliers / data entry errors
  - realistic class imbalance in the default label

Also generates 12 months of credit utilization history per applicant,
needed for the lagged utilization feature (3/6/12 month rolling averages).

Output:
  data/raw/credit_applications.csv
  data/raw/credit_utilization_history.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
N = 15000

def generate():
    app_ids = np.arange(100000, 100000 + N)

    # Application dates spanning 3 years, with a deliberate "policy change" midpoint
    start = datetime(2022, 1, 1)
    days_range = (datetime(2025, 1, 1) - start).days
    app_dates = [start + timedelta(days=int(d)) for d in RNG.integers(0, days_range, N)]

    age = RNG.integers(21, 70, N).astype(float)

    # --- Structural break: income reporting convention changed partway through ---
    # Before 2023-06-01: income reported ANNUALLY
    # After 2023-06-01: income reported MONTHLY
    # Neither is labeled in the raw column -- this must be detected/handled downstream.
    base_annual_income = RNG.lognormal(mean=10.8, sigma=0.45, size=N)  # ~ realistic annual salary distribution
    income_reported = np.where(
        np.array(app_dates) < datetime(2023, 6, 1),
        base_annual_income,
        base_annual_income / 12.0
    )

    credit_score = RNG.normal(680, 75, N).clip(300, 850)

    existing_debt = RNG.lognormal(mean=8.5, sigma=1.0, size=N)
    loan_amount = RNG.lognormal(mean=9.2, sigma=0.6, size=N)

    employment_years = RNG.exponential(scale=5, size=N).clip(0, 40)

    # Loan default outcome — realistic imbalance (~6-8% default rate)
    # Driven by a latent risk score plus noise
    risk_score = (
        -0.00002 * base_annual_income
        + 0.6 * (existing_debt / (base_annual_income + 1))
        - 0.01 * (credit_score - 680)
        - 0.05 * employment_years
        + RNG.normal(0, 1, N)
    )
    default_prob = 1 / (1 + np.exp(-(risk_score - 2.2)))
    defaulted = RNG.binomial(1, default_prob.clip(0.01, 0.6))

    df = pd.DataFrame({
        "application_id": app_ids,
        "application_date": app_dates,
        "age": age,
        "reported_income": income_reported,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "loan_amount_requested": loan_amount,
        "employment_years": employment_years,
        "defaulted": defaulted,
    })

    # --- Inject missingness (not random -- correlated with age/employment segment) ---
    missing_mask_income = (df["employment_years"] < 1) & (RNG.random(N) < 0.4)
    df.loc[missing_mask_income, "reported_income"] = np.nan

    missing_mask_credit = RNG.random(N) < 0.03
    df.loc[missing_mask_credit, "credit_score"] = np.nan

    missing_mask_employment = RNG.random(N) < 0.05
    df.loc[missing_mask_employment, "employment_years"] = np.nan

    # --- Inject data entry errors / outliers ---
    error_idx = RNG.choice(N, size=int(N * 0.01), replace=False)
    df.loc[error_idx, "age"] = df.loc[error_idx, "age"] * -1  # negative ages, obviously wrong

    error_idx2 = RNG.choice(N, size=int(N * 0.005), replace=False)
    df.loc[error_idx2, "credit_score"] = 999  # impossible sentinel value some systems use

    # --- Inconsistent date formatting (simulate multiple source systems) ---
    def format_date_inconsistently(d, i):
        if i % 3 == 0:
            return d.strftime("%Y-%m-%d")
        elif i % 3 == 1:
            return d.strftime("%d/%m/%Y")
        else:
            return d.strftime("%m-%d-%Y")

    df["application_date"] = [
        format_date_inconsistently(d, i) for i, d in enumerate(df["application_date"])
    ]

    # Shuffle rows so the structural break isn't trivially visible by row order
    shuffle_idx = RNG.permutation(N)
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    risk_score = risk_score[shuffle_idx]

    return df, risk_score


def generate_utilization_history(applications_df, risk_score):
    """
    12 months of credit utilization (% of limit used) per applicant.
    Trend is driven by the underlying continuous risk score (not the binary
    label directly) plus heavy noise, so it correlates with default risk
    without perfectly encoding it.
    """
    rows = []

    for i, (_, app) in enumerate(applications_df.iterrows()):
        app_id = app["application_id"]
        base_utilization = RNG.uniform(10, 60)

        # weak, noisy signal: higher risk score -> more likely to trend up,
        # but far from deterministic
        personal_drift_strength = np.clip(risk_score[i], -1, 3) * RNG.uniform(0.2, 0.8)

        for months_ago in range(12, 0, -1):
            drift = (12 - months_ago) * personal_drift_strength
            utilization = base_utilization + drift + RNG.normal(0, 10)
            utilization = np.clip(utilization, 0, 100)

            rows.append({
                "application_id": app_id,
                "months_ago": months_ago,
                "utilization_pct": round(utilization, 2),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, risk_score = generate()
    out_path = "data/raw/credit_applications.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Default rate: {df['defaulted'].mean():.2%}")
    print(f"Missing reported_income: {df['reported_income'].isna().mean():.2%}")
    print(f"Missing credit_score: {df['credit_score'].isna().mean():.2%}")

    history_df = generate_utilization_history(df, risk_score)
    history_path = "data/raw/credit_utilization_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Generated {len(history_df)} rows -> {history_path}")
