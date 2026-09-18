-- Rolling average utilization over 3/6/12 months per applicant.

with history as (

    select * from {{ ref('credit_utilization_history') }}

),

lagged as (

    select
        application_id,
        avg(case when months_ago <= 3 then utilization_pct end) as utilization_3mo_avg,
        avg(case when months_ago <= 6 then utilization_pct end) as utilization_6mo_avg,
        avg(case when months_ago <= 12 then utilization_pct end) as utilization_12mo_avg

    from history
    group by application_id

)

select * from lagged
