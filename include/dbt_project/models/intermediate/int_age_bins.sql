-- Buckets age into ranges for WoE scoring.

with staged as (

    select * from {{ ref('stg_credit_applications') }}

),

binned as (

    select
        application_id,
        age,
        defaulted,
        case
            when age is null then 'unknown'
            when age < 26 then '18-25'
            when age < 36 then '26-35'
            when age < 51 then '36-50'
            else '51-70'
        end as age_bin

    from staged

)

select * from binned
