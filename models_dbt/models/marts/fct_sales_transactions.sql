with normalized_sales as (
    select * from {{ ref('int_sales_currency_normalized') }}
)
select
    md5(concat(transaction_id, '_', line_item_id)) as transaction_line_id,
    transaction_id,
    sku_id,
    customer_id,
    date_key,
    store_id,
    province_code,
    transaction_timestamp,
    quantity_sold,
    unit_selling_price_usd,
    unit_selling_price_khr,
    applied_exchange_rate,
    payment_currency,
    khqr_merchant_mcc,
    payment_channel,
    gross_revenue_usd,
    cogs_usd,
    round(gross_revenue_usd - cogs_usd, 2) as net_profit_margin_usd
from normalized_sales
