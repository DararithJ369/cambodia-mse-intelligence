-- Mart: Inventory Turnover & Stock Velocity
-- Computes COGS, inventory valuation, 90-day turnover rates, and Days Sales of Inventory (DSI)
with sku_cogs as (
    select
        sku_id,
        sum(quantity_sold) as total_units_sold_90d,
        sum(gross_revenue_usd) as total_revenue_90d,
        sum(cogs_usd) as total_cogs_90d
    from {{ ref('fct_sales_transactions') }}
    group by sku_id
)
select
    p.sku_id,
    p.product_name_khmer,
    p.category_id,
    p.unit_cost_usd,
    p.unit_selling_price_usd,
    p.current_stock_on_hand,
    p.reorder_point_units,
    p.lead_time_days,
    round(p.current_stock_on_hand * p.unit_cost_usd, 2) as current_inventory_value_usd,
    coalesce(c.total_units_sold_90d, 0) as total_units_sold_90d,
    round(coalesce(c.total_cogs_90d, 0.0), 2) as total_cogs_90d,
    round(coalesce(c.total_revenue_90d, 0.0), 2) as total_revenue_90d,
    -- 90-Day Inventory Turnover Rate = Total COGS / Current Inventory Value
    round(c.total_cogs_90d / nullif(p.current_stock_on_hand * p.unit_cost_usd, 0), 2) as inventory_turnover_90d,
    -- Annualized Turnover Rate = Turnover 90d * (365 / 90)
    round((c.total_cogs_90d / nullif(p.current_stock_on_hand * p.unit_cost_usd, 0)) * (365.0 / 90.0), 2) as annualized_turnover_rate,
    -- Days Sales of Inventory (DSI) = 90 / Turnover 90d
    round(90.0 / nullif(c.total_cogs_90d / nullif(p.current_stock_on_hand * p.unit_cost_usd, 0), 0), 1) as days_sales_of_inventory_dsi
from {{ ref('dim_products') }} p
left join sku_cogs c on p.sku_id = c.sku_id
order by annualized_turnover_rate desc
