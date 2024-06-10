select date_trunc(day, install_time) dt,
        count(distinct
            case
            when campaign_name in ('web_app_install_large_overlay_deals', 'web_app_install_large_overlay_deals_dach')
                then app_user_id
            else null end
            ) as overlay_installs,
       count(distinct app_user_id) as total_installs
from stage.reporting_marketing.appsflyer_installs_uninstalls
where
    activity_kind = 'install'
    and install_time > '2023-10-01'
group by 1
order by 1;