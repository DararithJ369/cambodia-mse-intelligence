with stg_sales as (
    select * from {{ ref('stg_pos_sales') }}
),
stg_products as (
    select sku_id, unit_cost_usd from {{ ref('stg_inventory') }}
),
stg_payments as (
    select transaction_id, khqr_merchant_mcc, payment_channel from {{ ref('stg_khqr_payments') }}
)
select
    s.transaction_id,
    s.line_item_id,
    s.sku_id,
    s.customer_id,
    s.date_key,
    s.transaction_timestamp,
    s.store_id,
    s.province_code,
    s.quantity_sold,
    s.unit_selling_price_usd,
    s.unit_selling_price_khr,
    s.applied_exchange_rate,
    s.payment_currency,
    coalesce(p.khqr_merchant_mcc, '5411') as khqr_merchant_mcc,
    coalesce(p.payment_channel, 'CASH')   as payment_channel,
    
    -- Normalize Gross Revenue to USD
    case 
        when s.payment_currency = 'KHR' 
            then (s.quantity_sold * s.unit_selling_price_khr) / nullif(s.applied_exchange_rate, 0)
        else (s.quantity_sold * s.unit_selling_price_usd)
    end as gross_revenue_usd,

    -- Calculate COGS using inventory unit cost
    (s.quantity_sold * coalesce(pr.unit_cost_usd, 0.0)) as cogs_usd

from stg_sales s
left join stg_products pr on s.sku_id = pr.sku_id
left join stg_payments p on s.transaction_id = p.transaction_id
