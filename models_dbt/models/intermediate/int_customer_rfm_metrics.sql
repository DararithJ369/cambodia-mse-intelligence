with sales as (
    select * from {{ ref('stg_pos_sales') }}
),
max_date_cte as (
    select max(date_key) as max_date from sales
),
customer_aggregates as (
    select
        s.customer_id,
        count(distinct s.transaction_id) as frequency_count_180d,
        sum(s.quantity_sold * s.unit_selling_price_usd) as monetary_total_usd,
        max(s.date_key) as last_transaction_date,
        -- Pick top payment method
        mode(s.payment_method) as preferred_payment_channel
    from sales s
    where s.customer_id != 'CUST-ANON-0000'
    group by s.customer_id
),
rfm_calculated as (
    select
        ca.customer_id,
        ca.frequency_count_180d,
        round(ca.monetary_total_usd, 2) as monetary_total_usd,
        ca.last_transaction_date,
        ca.preferred_payment_channel,
        datediff('day', ca.last_transaction_date, m.max_date) as recency_days
    from customer_aggregates ca
    cross join max_date_cte m
)
select
    customer_id,
    recency_days,
    frequency_count_180d,
    monetary_total_usd,
    preferred_payment_channel,
    case
        when recency_days <= 14 and frequency_count_180d >= 15 then 'Champions'
        when recency_days <= 30 and frequency_count_180d >= 10 then 'Loyal Customers'
        when recency_days <= 30 and frequency_count_180d < 10  then 'Potential Loyalists'
        when recency_days > 30 and frequency_count_180d >= 8   then 'At Risk'
        else 'Occasional / Hibernating'
    end as rfm_segment_cluster,
    round(least(1.0, greatest(0.0, recency_days / 90.0)), 2) as churn_probability_score,
    round(monetary_total_usd * 1.5, 2) as predicted_clv_usd
from rfm_calculated
