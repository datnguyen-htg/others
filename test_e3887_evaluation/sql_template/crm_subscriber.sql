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
    and SESSION_TIMESTAMP >= '2024-02-15'
    and DEVICE_TYPE = 'Mobile'
    ),

    users as (
    select
        ID,
--         IS_SUBSCRIBER,
--         IS_ACCOUNT,
--         IS_CUSTOMER,
        ORIGIN_SOURCE,
--         ORIGIN_VARIATION,
        c.uref
    from raw_ab_traffic a
    left join stage.derived.sessions s on a.session_id = s.session_id
    left join stage.derived_marketing.crm_users c on s.uref = c.uref
    where c.REGISTRATION_DATE >= '2024-02-15'
    and c.IS_SUBSCRIBER = TRUE
    )

select
*
from users
PIVOT ( count( uref) for id in ('e3887v0', 'e3887v1') )
as p(origin_source, "'e3887v0'","'e3887v1'" )
order by 1
;