with raw_sales as (
    select distinct 
        transaction_id,
        timestamp,
        date,
        payment_method
    from {{ source('ods_pos', 'raw_transactions') }}
)
select
    cast(transaction_id as varchar(64))  as transaction_id,
    cast(timestamp as timestamp)         as payment_timestamp,
    cast(date as date)                   as date_key,
    cast(payment_method as varchar(32))  as raw_payment_method,
    case
        when payment_method = 'KHQR_ABA' then 'ABA_KHQR'
        when payment_method = 'KHQR_Bakong' then 'BAKONG_KHQR'
        when payment_method in ('Cash_USD', 'Cash_KHR') then 'CASH'
        when payment_method = 'Debit_Credit_Card' then 'CARD'
        else 'OTHER'
    end                                  as payment_channel,
    case 
        when payment_method in ('KHQR_ABA', 'KHQR_Bakong') then true 
        else false 
    end                                  as is_digital_qr_payload,
    '5411'                               as khqr_merchant_mcc,
    case 
        when payment_method in ('Cash_KHR', 'KHQR_Bakong') then 'KHR'
        else 'USD'
    end                                  as settlement_currency
from raw_sales
