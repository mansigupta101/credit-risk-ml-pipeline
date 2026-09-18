# Credit Risk ML Pipeline

An end-to-end credit risk pipeline: synthetic loan data through Airflow, cleaned and feature-engineered in dbt, modeled with XGBoost/Logistic Regression + SHAP, evaluated in dollar terms, monitored for drift, and served via FastAPI.

Built to show the infrastructure around a model, not just the model.

## Pipeline

```
raw data -> Airflow (ingest + validate) -> dbt (staging -> features -> mart)
   -> model training (LogReg + XGBoost + SHAP)
   -> financial impact + drift monitoring
   -> FastAPI (Docker)
```

## Results

- XGBoost: ROC-AUC 0.67, PR-AUC 0.13
- Logistic Regression: ROC-AUC 0.71, PR-AUC 0.19
- At a 0.5 threshold: 39% less bad debt vs. approving everyone, 77% of good customers still approved
- Drift check flags a simulated downturn (utilization PSI 0.66), stays quiet where nothing changed (credit score PSI 0.0)


## Structure

```
dags/                   Airflow DAG - fetch, validate, trigger dbt
include/dbt_project/    dbt models (staging -> intermediate -> marts)
ml/                      training + financial evaluation
monitoring/              drift detection (Evidently)
api/                     FastAPI app + Dockerfile
data/                    synthetic data generator
```

## Running it

```
astro dev start                        # Airflow + dbt, localhost:8080
python3 ml/train_model.py
python3 ml/financial_evaluation.py
python3 monitoring/drift_check.py
docker build -f api/Dockerfile -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api    # localhost:8000
```

## Notes for anyone reviewing this

- Used DuckDB instead of Postgres/Snowflake for local dev
- The synthetic data keeps default outcomes fairly noisy on purpose. Real credit models typically land around 0.65-0.75 AUC too
- Decision threshold is 0.5, just a placeholder. Not tuned to real cost trade-offs
- Great Expectations checks currently just log problems instead of stopping the pipeline
- Logistic Regression throws overflow warnings from a few outlier rows in the raw data. Suppressed, not fixed

## Stack

Airflow (Astro CLI) · dbt + DuckDB · Great Expectations · scikit-learn · XGBoost · SHAP · Evidently · FastAPI · Docker
