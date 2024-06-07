with
    sessions_in_scope as (
        select
            session_id,
            session_timestamp,
            ID,
            cookie_user_id

        from STAGE.DERIVED_AB_TESTS.AB_TESTS_USERS_BASE_SESSIONS_W_MARKERS
        where TEST_ID = 'e3887'
        and SESSION_TIMESTAMP >= '2024-02-15'
        and DEVICE_TYPE = 'Mobile'
    ),

    overlay_cta_clicks as (
        SELECT session_id,
               metric_name,
               INTERACTIONS_KEY
        FROM stage.reporting_product.fct_interactions
        WHERE interaction_timestamp >= '2024-02-15'
            AND se_category = 'app_install_large_overlay'
            AND metric_name = 'app_install_large_overlay_deeplink_cta_click'
            and session_id in (select distinct session_id from sessions_in_scope)
        GROUP BY all
    )
select
    ID,
--     date_trunc('day', session_timestamp) dt,
    count(distinct s.session_id) as sessions,
    count(distinct o.INTERACTIONS_KEY) as app_install_large_overlay
from sessions_in_scope s
left join overlay_cta_clicks o using (session_id)
group by all
order by 1, 2;