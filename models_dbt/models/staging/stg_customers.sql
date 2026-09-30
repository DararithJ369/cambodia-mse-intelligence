with raw_cust as (
    select * from {{ source('ods_pos', 'raw_customers') }}
)
select
    cast(customer_id as varchar(64))         as customer_id,
    cast(customer_name as varchar(128))      as customer_name_khmer,
    cast(phone_number as varchar(32))        as phone_number,
    cast(loyalty_tier as varchar(32))        as loyalty_tier,
    cast(primary_city as varchar(64))        as province_code,
    cast(account_created_date as date)       as registration_date
from raw_cust
