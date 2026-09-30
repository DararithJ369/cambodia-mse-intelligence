with raw_fx as (
    select * from {{ source('ods_pos', 'raw_fx_rates') }}
)
select
    cast(rate_date as date)                      as date_key,
    cast(nbc_usd_khr_rate as decimal(8,2))       as nbc_official_rate,
    cast(bank_buy_rate as decimal(8,2))          as bank_buy_rate,
    cast(bank_sell_rate as decimal(8,2))         as bank_sell_rate
from raw_fx
