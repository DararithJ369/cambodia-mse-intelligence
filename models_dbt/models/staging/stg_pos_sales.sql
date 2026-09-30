with raw_sales as (
    select * from {{ source('ods_pos', 'raw_transactions') }}
)
select
    cast(transaction_id as varchar(64))                  as transaction_id,
    cast(item_line_no as integer)                        as line_item_id,
    cast(sku as varchar(64))                             as sku_id,
    cast(customer_id as varchar(64))                    as customer_id,
    cast(timestamp as timestamp)                         as transaction_timestamp,
    cast(date as date)                                   as date_key,
    cast(store_id as varchar(32))                        as store_id,
    cast(store_city as varchar(64))                      as province_code,
    cast(quantity as integer)                            as quantity_sold,
    cast(unit_price_usd as decimal(10,2))                as unit_selling_price_usd,
    cast(unit_price_khr as decimal(12,2))                as unit_selling_price_khr,
    cast(nbc_exchange_rate as decimal(8,2))              as applied_exchange_rate,
    cast(payment_method as varchar(32))                  as payment_method,
    case 
        when payment_method in ('Cash_KHR', 'KHQR_Bakong') then 'KHR'
        else 'USD'
    end                                                  as payment_currency
from raw_sales
