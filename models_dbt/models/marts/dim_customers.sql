with customers as (
    select * from {{ ref('stg_customers') }}
),
rfm as (
    select * from {{ ref('int_customer_rfm_metrics') }}
)
select
    c.customer_id,
    c.customer_name_khmer,
    c.phone_number,
    c.province_code,
    c.loyalty_tier,
    c.registration_date,
    coalesce(r.recency_days, 999)                            as recency_days,
    coalesce(r.frequency_count_180d, 0)                     as frequency_count_180d,
    coalesce(r.monetary_total_usd, 0.0)                      as monetary_total_usd,
    coalesce(r.preferred_payment_channel, 'CASH')            as preferred_payment_channel,
    coalesce(r.rfm_segment_cluster, 'Occasional / Hibernating') as rfm_segment_cluster,
    coalesce(r.churn_probability_score, 1.0)                 as churn_probability_score,
    coalesce(r.predicted_clv_usd, 0.0)                       as predicted_clv_usd
from customers c
left join rfm r on c.customer_id = r.customer_id
