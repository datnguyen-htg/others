select date_trunc(day, install_time) dt,
       count(distinct app_user_id) as installs
from stage.reporting_marketing.appsflyer_installs_uninstalls
where
    activity_kind = 'install'
    and install_time > '2023-12-01'
group by 1
order by 1;