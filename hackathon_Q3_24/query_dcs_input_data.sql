select *
from stage.reporting_finance.dcs_input_data
where true
and site_global_region = 'DACH'
and site_id= 'hometogo.de'
and INTER_COMPANY_GROUP = 'FALSE'
and DEAL_TYPE ilike '%onsite%'