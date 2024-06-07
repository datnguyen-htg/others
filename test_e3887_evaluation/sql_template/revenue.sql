
select
    date_trunc('day', SESSION_TIMESTAMP) dt,
	ID,
	count(distinct cookie_user_id) users,
	count(distinct session_id) sessions,
	sum(revenue) revenues,
    sum(booking_quantity_direct_before_cancellations_onsite_booking) onsite_bookings,
    sum(searches) searches,
    count(distinct iff(booking_quantity_direct_before_cancellations_onsite_booking>0,cookie_user_id, null)) as UWOB,

    sum(users) over(partition by ID order by DT asc) cum_users,
    sum(revenues) over(partition by ID order by DT asc) cum_revenues,
    sum(onsite_bookings) over(partition by ID order by DT asc) cum_onsite_bookings,
    sum(UWOB) over(partition by ID order by DT asc) cum_UWOB,

    cum_revenues/cum_users RPU,
    cum_onsite_bookings/cum_users onsite_booking_per_user
from STAGE.DERIVED_AB_TESTS.AB_TESTS_USERS_BASE_SESSIONS_W_MARKERS
where TEST_ID = 'e3887'
and SESSION_TIMESTAMP >= '2024-01-01'
and DEVICE_TYPE = 'Mobile'
group by 1,2
order by 1,2
