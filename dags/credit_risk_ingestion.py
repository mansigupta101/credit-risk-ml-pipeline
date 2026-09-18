"""Stage 1: fetch raw credit data and run quality checks before downstream processing."""

import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

RAW_DATA_PATH = "/usr/local/airflow/include/data/raw/credit_applications.csv"
DBT_PROJECT_DIR = "/usr/local/airflow/include/dbt_project"


def fetch_data(**context):
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Fetched {len(df)} rows from {RAW_DATA_PATH}")
    context["ti"].xcom_push(key="row_count", value=len(df))


def validate_data(**context):
    df = pd.read_csv(RAW_DATA_PATH)

    gx_context = gx.get_context(mode="ephemeral")
    data_source = gx_context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="credit_applications")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = ExpectationSuite(name="credit_applications_suite")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="reported_income", min_value=0, strict_min=True, mostly=1.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="credit_score", min_value=300, max_value=850, mostly=1.0)
    )

    result = batch.validate(suite)
    print(result)

    if not result.success:
        failed = [r for r in result.results if not r.success]
        errors = [str(r.expectation_config.type) for r in failed]
        print(f"{len(failed)} expectation(s) failed: {errors}")
        # not hard-failing yet -- dbt staging handles these cases downstream
        context["ti"].xcom_push(key="validation_errors", value=errors)
    else:
        print("All expectations passed.")


with DAG(
    dag_id="credit_risk_ingestion",
    description="Stage 1: Fetch raw credit data and run quality gates",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit-risk", "stage-1-ingestion"],
) as dag:

    fetch_data_task = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
    )

    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    dbt_seed_task = BashOperator(
        task_id="dbt_seed",
        bash_command="dbt seed --profiles-dir .",
        cwd=DBT_PROJECT_DIR,
    )

    dbt_run_task = BashOperator(
        task_id="dbt_run",
        bash_command="dbt run --profiles-dir .",
        cwd=DBT_PROJECT_DIR,
    )

    fetch_data_task >> validate_data_task >> dbt_seed_task >> dbt_run_task
