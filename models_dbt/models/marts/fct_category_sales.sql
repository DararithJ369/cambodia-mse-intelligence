-- Mart: Category Sales & Merchandise Performance
with cat_sales as (
    select
        p.category_id,
        count(distinct f.transaction_id) as checkout_baskets_count,
        count(distinct f.sku_id) as active_skus_count,
        sum(f.quantity_sold) as total_units_sold,
        sum(f.gross_revenue_usd) as category_gross_revenue_usd,
        sum(f.cogs_usd) as category_cogs_usd,
        sum(f.net_profit_margin_usd) as category_net_profit_usd
    from {{ ref('fct_sales_transactions') }} f
    join {{ ref('dim_products') }} p on f.sku_id = p.sku_id
    group by p.category_id
),
totals as (
    select sum(category_gross_revenue_usd) as grand_total_revenue from cat_sales
)
select
    c.category_id,
    c.active_skus_count,
    c.checkout_baskets_count,
    c.total_units_sold,
    round(c.category_gross_revenue_usd, 2) as category_gross_revenue_usd,
    round(c.category_cogs_usd, 2) as category_cogs_usd,
    round(c.category_net_profit_usd, 2) as category_net_profit_usd,
    round((c.category_net_profit_usd * 100.0 / nullif(c.category_gross_revenue_usd, 0)), 2) as gross_margin_pct,
    round((c.category_gross_revenue_usd * 100.0 / t.grand_total_revenue), 2) as revenue_share_pct
from cat_sales c
cross join totals t
order by category_gross_revenue_usd desc
