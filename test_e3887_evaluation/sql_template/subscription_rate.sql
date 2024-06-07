with
    raw_ab_traffic as (
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

    users as (
    select
        a.ID,
        date_trunc('day', a.session_timestamp) as dt,
        a.session_id,
        a.cookie_user_id,
        case
            when (c.REGISTRATION_DATE >= '2024-02-15' and c.IS_SUBSCRIBER = TRUE) then c.uref
            else null
        end uref
    from raw_ab_traffic a
    left join stage.derived.sessions s on a.session_id = s.session_id
    left join stage.derived_marketing.crm_users c on s.uref = c.uref
    )

select
    ID,
    dt,
    count(cookie_user_id) as cookie_user_ids,
    count(uref) as subscribers,
    sum(cookie_user_ids) over(partition by ID order by dt) as cum_users,
    sum(subscribers) over(partition by ID order by dt) as cum_subscribers,
    cum_subscribers/cum_users as subscription_rate
from users
group by all
order by 1, 2