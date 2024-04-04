with
troas as (
    select distinct
          BIDDING_STRATEGY_ID as BID_STRATEGY_ID,
          BIDDING_STRATEGY_TYPE

   from stage.main_marketing.MV_SEM_PERFORMANCE_REPORTS_GOOGLE_AD_GROUPS_TROAS
   where mkt_date >= '2022-01-01'
        and mkt_date < '2023-03-01'

),

bugdet as (
    select
        mkt_date,
        id ,
        mkt_account_id,
        BID_STRATEGY_ID,
        day_budget,
        RECOMMENDED_BUDGET_AMOUNT,
        CONVERSIONS_VALUE,
        mkt_cost,
        conversions
    from stage.main_marketing.MV_SEM_PERFORMANCE_REPORTS_GOOGLE_BUDGETS
    where mkt_date >= '2022-01-01'
        and mkt_date < '2023-03-01'


),

cte_final as (
    select b.MKT_DATE,
         b.mkt_account_id,
         b.id,
         t.BID_STRATEGY_ID,
--          t.BIDDING_STRATEGY_TARGET_ROAS,
         t.BIDDING_STRATEGY_TYPE,
--          t.AD_GROUP_TARGET_ROAS,
         b.day_budget,
         b.RECOMMENDED_BUDGET_AMOUNT,
         b.CONVERSIONS_VALUE,
         b.mkt_cost,
         b.conversions,
         b.CONVERSIONS_VALUE / nullif(b.mkt_cost,0) as ROAS
    from bugdet b
           left join troas t on t.BID_STRATEGY_ID = b.BID_STRATEGY_ID
  )
select
   *
from cte_final
;