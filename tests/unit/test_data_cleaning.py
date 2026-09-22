"""
Script to test the dbt cleaning logic by querying the tables in DuckDB.

NOTE: Requires `dbt run` to have been executed first (include/dbt_project) 
before executing this script.
"""

import duckdb
import pytest

DB_PATH = "include/dbt_project/dev.duckdb"


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect(DB_PATH, read_only=True)
    yield connection
    connection.close()


def test_no_sentinel_credit_scores(con):
    """999 sentinel values should be nulled out, not left as fake scores."""
    result = con.execute(
        "select count(*) from stg_credit_applications where credit_score = 999"
    ).fetchone()
    assert result[0] == 0


def test_credit_score_within_valid_range(con):
    """Real credit scores must fall between 300 and 850."""
    result = con.execute(
        "select count(*) from stg_credit_applications "
        "where credit_score is not null and (credit_score < 300 or credit_score > 850)"
    ).fetchone()
    assert result[0] == 0


def test_no_negative_ages(con):
    """Negative ages were a known data entry error -- should be corrected."""
    result = con.execute(
        "select count(*) from stg_credit_applications where age < 0"
    ).fetchone()
    assert result[0] == 0


def test_application_dates_parsed(con):
    """All three raw date formats should parse to valid, non-null dates."""
    result = con.execute(
        "select count(*) from stg_credit_applications where application_date is null"
    ).fetchone()
    assert result[0] == 0


def test_income_structural_break_corrected(con):
    """
    Pre-2023-06-01 income was annual, post was monthly in the raw data.
    After correction, annual_income should be in a plausible salary range
    across the whole date span, not show a jump at the break date.
    """
    result = con.execute(
        """
        select
            avg(case when application_date < '2023-06-01' then annual_income end) as pre_break_avg,
            avg(case when application_date >= '2023-06-01' then annual_income end) as post_break_avg
        from stg_credit_applications
        where annual_income is not null
        """
    ).fetchone()
    pre_break_avg, post_break_avg = result
    # both sides should be in a similar, plausible salary range -- not off by ~12x
    ratio = max(pre_break_avg, post_break_avg) / min(pre_break_avg, post_break_avg)
    assert ratio < 1.5


def test_mart_has_no_duplicate_applications(con):
    """Each application_id should appear exactly once in the final feature table."""
    result = con.execute(
        "select count(*) - count(distinct application_id) from mart_credit_features"
    ).fetchone()
    assert result[0] == 0
