with inventory as (
    select * from {{ ref('stg_inventory') }}
)
select
    sku_id,
    product_name_khmer,
    category_id,
    unit_cost_usd,
    round(unit_cost_usd * 4085 / 100) * 100 as unit_cost_khr,
    unit_selling_price_usd,
    current_stock_on_hand,
    allocated_stock,
    reorder_point_units,
    safety_stock_level,
    lead_time_days,
    shelf_life_days,
    supplier_name,
    case
        when current_stock_on_hand <= reorder_point_units then true
        else false
    end as is_reorder_required
from inventory
