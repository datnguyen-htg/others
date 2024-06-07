with
    raw_ab_traffic as (
    select
        session_id,
        session_timestamp,
        ID,
        cookie_user_id,
        revenue,
        booking_quantity_direct_before_cancellations,
        searches
    from STAGE.DERIVED_AB_TESTS.AB_TESTS_USERS_BASE_SESSIONS_W_MARKERS
    where TEST_ID = 'e3887'
    and SESSION_TIMESTAMP >= '2024-01-01'
    and DEVICE_TYPE = 'Mobile'
    ),

    raw_crm_modal as (
    select
        session_id,
        account_subscription_apple,
        account_subscription_facebook,
        account_subscription_google,
        account_subscription_email,
        account_creation_apple,
        account_creation_facebook,
        account_creation_google,
        account_creation_email,
        account_subscription_apple + account_subscription_facebook + account_subscription_google + account_subscription_email account_subscription,
        account_creation_apple + account_creation_facebook + account_creation_google + account_creation_email account_creation
    from stage.derived_product.ab_diagnostics_crm_modals
    where test_ids::varchar like '%e3887%'
    and session_timestamp >= '2024-01-01'
    ),

    cte_join as (
    select a.*,
           m.account_subscription,
           m.account_creation
    from raw_ab_traffic a
    left join raw_crm_modal m using (session_id)
    ),

    cte_final as (
    select
        date_trunc('day', SESSION_TIMESTAMP) dt,
	    ID,
        count(distinct cookie_user_id) users,
        sum(account_subscription) account_subscriptions,
	    sum(account_creation) account_creations,
        sum(users) over(partition by ID order by DT asc) cum_users,
        sum(account_subscriptions) over(partition by ID order by DT asc) cum_account_subcription,
        sum(account_creations) over(partition by ID order by DT asc) cum_account_creation,
        cum_account_subcription/cum_users subcription_rate,
        cum_account_creation/cum_users creation_rate
    from cte_join
    group by all
    )

select * from cte_final