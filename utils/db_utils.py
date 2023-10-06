import os
import pandas as pd

import snowflake.connector

from utils.config import DEFAULT_USER_ENV_VAR, DEFAULT_ROLE_ENV_VAR, DEFAULT_PW_ENV_VAR, DEFAULT_ACCOUNT_ENV_VAR, DEFAULT_WAREHOUSE_ENV_VAR

def df_from_snowflake(query:str, user_env_var:str=None, role_env_var:str=None, password_env_var:str=None, account_env_var:str=None)->pd.DataFrame:
    _user_env_var=user_env_var or DEFAULT_USER_ENV_VAR
    _role_env_var=role_env_var or DEFAULT_ROLE_ENV_VAR
    _password_env_var=password_env_var or DEFAULT_PW_ENV_VAR
    _account_env_var=account_env_var or DEFAULT_ACCOUNT_ENV_VAR
    _warehouse_env_var=account_env_var or DEFAULT_WAREHOUSE_ENV_VAR

    credentials=dict(
        user=os.getenv(_user_env_var),
        role=os.getenv(_role_env_var),
        password=os.getenv(_password_env_var),
        account=os.getenv(_account_env_var),
        warehouse=os.getenv(_warehouse_env_var),
    )

    for k,v in credentials.items():
        if v is None:
            raise Exception('Env variable %s not set!' % k)

    df=None

    ctx = snowflake.connector.connect(**credentials)
    cs = ctx.cursor()
    try:
        cs.execute(query)
        df = cs.fetch_pandas_all()
    finally:
        cs.close()
    ctx.close()

    return df