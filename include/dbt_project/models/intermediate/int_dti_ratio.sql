-- Calculates debt-to-income ratio from cleaned application data.

with staged as (

    select * from {{ ref('stg_credit_applications') }}

),

with_dti as (

    select
        application_id,
        application_date,
        age,
        annual_income,
        credit_score,
        existing_debt,
        loan_amount_requested,
        employment_years,
        defaulted,

        -- debt-to-income ratio: existing debt divided by annual income
        -- capped at 10 to avoid extreme values from very low incomes
        case
            when annual_income > 0 then least(existing_debt / annual_income, 10)
            else null
        end as debt_to_income_ratio

    from staged

)

select * from with_dti
