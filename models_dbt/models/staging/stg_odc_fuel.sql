with raw_fuel as (
    select * from {{ source('ods_pos', 'raw_odc_fuel') }}
)
select
    cast(im_date as date)                   as date_key,
    cast(regu_gas as decimal(10,2))         as fuel_price_regular_khr,
    cast(diesel_gas as decimal(10,2))       as fuel_price_diesel_khr,
    cast(moc_no as varchar(32))             as moc_notification_number
from raw_fuel
