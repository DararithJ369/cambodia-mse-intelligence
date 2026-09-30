with fx_dates as (
    select * from {{ ref('stg_nbc_rates') }}
),
fuel as (
    select * from {{ ref('stg_odc_fuel') }}
)
select
    d.date_key,
    extract('year' from d.date_key)          as calendar_year,
    extract('month' from d.date_key)         as calendar_month,
    extract('day' from d.date_key)           as calendar_day,
    dayname(d.date_key)                      as day_of_week_name,
    case
        when dayofweek(d.date_key) in (0, 6) then true
        else false
    end                                      as is_weekend_flag,
    coalesce(f.fuel_price_regular_khr, 4800) as fuel_price_regular_khr,
    coalesce(f.fuel_price_diesel_khr, 5450)  as fuel_price_diesel_khr,
    d.nbc_official_rate,
    -- Benchmark provincial density from NIS 2019 Census for Phnom Penh vs provincial median
    3136.0                                   as provincial_pop_density,
    -- Major Cambodian holiday events in Q3 (Constitution Day Sep 24, Pchum Ben late Sep)
    case
        when strftime(d.date_key, '%m-%d') in ('09-24', '09-27', '09-28', '09-29') then true
        else false
    end                                      as holiday_event_flag,
    -- Simulated seasonal monsoon rainfall (mm) for Cambodia rainy season (July - Sept)
    round(cast(random() * 25.0 + 5.0 as decimal(5,1)), 1) as rainfall_mm_local
from fx_dates d
left join fuel f on d.date_key = f.date_key
