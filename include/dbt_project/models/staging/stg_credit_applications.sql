-- Cleans raw credit application data before any feature engineering.

with source as (

    select * from {{ ref('credit_applications') }}

),

parsed_dates as (

    select
        application_id,

        -- dates come in 3 formats: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY
        coalesce(
            try_strptime(application_date, '%Y-%m-%d'),
            try_strptime(application_date, '%d/%m/%Y'),
            try_strptime(application_date, '%m-%d-%Y')
        )::date as application_date,

        age,
        reported_income,
        credit_score,
        existing_debt,
        loan_amount_requested,
        employment_years,
        defaulted

    from source

),

cleaned as (

    select
        application_id,
        application_date,

        -- some ages were recorded as negative, sign error only
        abs(age) as age,

        -- income was reported annually before 2023-06-01, monthly after
        case
            when application_date < date '2023-06-01' then reported_income
            else reported_income * 12
        end as annual_income,

        -- 999 is a sentinel for missing, not a real score
        case
            when credit_score = 999 then null
            else credit_score
        end as credit_score,

        -- capped at 50000/40000: raw data has extreme outliers from a long-tailed distribution
        least(existing_debt, 50000) as existing_debt,
        least(loan_amount_requested, 40000) as loan_amount_requested,
        employment_years,
        defaulted

    from parsed_dates

)

select * from cleaned
