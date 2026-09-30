with raw_inv as (
    select * from {{ source('ods_pos', 'raw_inventory') }}
)
select
    cast(sku as varchar(64))                             as sku_id,
    cast(product_name as varchar(255))                   as product_name_khmer,
    cast(category as varchar(64))                        as category_id,
    cast(cost_price_usd as decimal(10,2))                as unit_cost_usd,
    cast(selling_price_usd as decimal(10,2))             as unit_selling_price_usd,
    cast(current_stock_level as integer)                 as current_stock_on_hand,
    cast(round(current_stock_level * 0.15) as integer)   as allocated_stock,
    cast(reorder_point as integer)                       as reorder_point_units,
    cast(round(reorder_point * 0.6) as integer)          as safety_stock_level,
    cast(lead_time_days as integer)                      as lead_time_days,
    case
        when category in ('Dairy & Cold', 'Fresh Essentials') then 14
        when category in ('Beverages', 'Packaged Foods', 'Snacks') then 180
        else 730
    end                                                  as shelf_life_days,
    cast(supplier_name as varchar(128))                  as supplier_name
from raw_inv
