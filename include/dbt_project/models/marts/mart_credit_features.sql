-- Final feature table for model training. Joins DTI, lagged utilization,
-- and age WoE onto the cleaned base data.

{{ config(materialized='table') }}

with base as (

    select * from {{ ref('int_dti_ratio') }}

),

lagged_util as (

    select * from {{ ref('int_lagged_utilization') }}

),

age_bins as (

    select * from {{ ref('int_age_bins') }}

),

age_woe as (

    select * from {{ ref('int_age_woe') }}

),

joined as (

    select
        base.application_id,
        base.application_date,
        base.age,
        base.annual_income,
        base.credit_score,
        base.existing_debt,
        base.loan_amount_requested,
        base.employment_years,
        base.debt_to_income_ratio,

        lagged_util.utilization_3mo_avg,
        lagged_util.utilization_6mo_avg,
        lagged_util.utilization_12mo_avg,

        age_bins.age_bin,
        age_woe.age_woe,

        base.defaulted

    from base
    left join lagged_util on base.application_id = lagged_util.application_id
    left join age_bins on base.application_id = age_bins.application_id
    left join age_woe on age_bins.age_bin = age_woe.age_bin

)

select * from joined
