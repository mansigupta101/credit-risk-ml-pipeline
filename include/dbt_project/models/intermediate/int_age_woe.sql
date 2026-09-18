-- WoE per age bucket: ln(share of good applicants / share of bad applicants).
-- Positive WoE = safer bucket, negative WoE = riskier bucket.

with binned as (

    select * from {{ ref('int_age_bins') }}

),

bucket_counts as (

    select
        age_bin,
        count(*) as total,
        sum(defaulted) as bad_count,
        count(*) - sum(defaulted) as good_count

    from binned
    group by age_bin

),

totals as (

    select
        sum(bad_count) as total_bad,
        sum(good_count) as total_good

    from bucket_counts

),

woe as (

    select
        bucket_counts.age_bin,
        bucket_counts.total,
        bucket_counts.bad_count,
        bucket_counts.good_count,
        ln(
            (bucket_counts.good_count::float / totals.total_good)
            / nullif(bucket_counts.bad_count::float / totals.total_bad, 0)
        ) as age_woe

    from bucket_counts
    cross join totals

)

select * from woe
