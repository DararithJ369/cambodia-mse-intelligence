-- Mart: Daily Revenue & Gross Profit Margins
-- Computes daily sales metrics, COGS, net profit, and rolling 7-day margin trends
with daily_sales as (
    select
        date_key,
        count(distinct transaction_id) as total_checkout_baskets,
        count(*) as total_line_items_sold,
        sum(quantity_sold) as total_units_sold,
        sum(gross_revenue_usd) as daily_gross_revenue_usd,
        sum(unit_selling_price_khr * quantity_sold) as daily_gross_revenue_khr,
        sum(cogs_usd) as daily_cogs_usd,
        sum(net_profit_margin_usd) as daily_net_profit_usd
    from {{ ref('fct_sales_transactions') }}
    group by date_key
)
select
    date_key,
    total_checkout_baskets,
    total_line_items_sold,
    total_units_sold,
    round(daily_gross_revenue_usd, 2) as daily_gross_revenue_usd,
    round(daily_gross_revenue_khr, 0) as daily_gross_revenue_khr,
    round(daily_cogs_usd, 2) as daily_cogs_usd,
    round(daily_net_profit_usd, 2) as daily_net_profit_usd,
    round((daily_net_profit_usd * 100.0 / nullif(daily_gross_revenue_usd, 0)), 2) as gross_profit_margin_pct,
    round(avg(daily_gross_revenue_usd) over (
        order by date_key rows between 6 preceding and current row
    ), 2) as rolling_7d_avg_revenue_usd,
    round(avg(daily_net_profit_usd * 100.0 / nullif(daily_gross_revenue_usd, 0)) over (
        order by date_key rows between 6 preceding and current row
    ), 2) as rolling_7d_profit_margin_pct
from daily_sales
order by date_key
