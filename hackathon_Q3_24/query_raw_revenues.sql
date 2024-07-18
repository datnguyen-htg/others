with 

cte_dates as (
    select
        '2024-01-01'::date as start_date,
        current_date -2 as end_date
),

cte_dim_site as (
    select
        site_id                                                                                                             as dim_site_id,
        legal_entity                                                                                                        as dim_legal_entity
    from stage.reporting.dim_site as dim_site
), 

cte_revenue as (
    select
        date,
        conversion_click_id,
        iff(cte_dim_site.dim_legal_entity in ('casamundo'), 'hometogo', cte_dim_site.dim_legal_entity) as legal_entity,
        site_global_region,
        site_id,
        inter_company_group,
        deal_type,
        iff(deal_type <> 'Others - CPC & CPL', rev.conversions, 0) as booking_quantity,
        rev.est_revenue_before_cancellations                      as booking_revenues,
        rev.est_revenue                                           as booking_revenues_after_cancellations,
        iff(date_ifrs = date, rev.revenue,0)                        as ifrs_revenues
    from stage.reporting.main_revenues rev
    left join cte_dim_site on rev.site_id = cte_dim_site.dim_site_id
    where true
        and inter_company_group = false 
        and (legal_entity <> 'feries' or legal_entity is null) 
        and site_global_region <> 'Unknown'
        and date between (select start_date from cte_dates) and (select end_date from cte_dates)
        and site_global_region = 'DACH'
        and contains(site_id, '.de')
    -- group by 1,2,3,4,5,6
), 


 cte_final as (
    select
        cte_revenue.date,
        cte_revenue.site_id,
        cte_revenue.legal_entity,
        cte_revenue.site_global_region,
        cte_revenue.inter_company_group,
        cte_revenue.deal_type,
        cte_revenue.booking_quantity,
        cte_revenue.booking_revenues,
        cte_revenue.booking_revenues_after_cancellations,
        cte_revenue.ifrs_revenues
    from cte_revenue
    order by 1,2,3
)

select 
	*
from cte_final
where date between (select start_date from cte_dates) and (select end_date from cte_dates)