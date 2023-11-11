#%%
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
#%%
def get_erc20_tokens_prices(engine):
    utc_now = datetime.utcnow()
    start_date = utc_now - timedelta(days = 5)
    sql_cmd_1 = '''
    SELECT * FROM erc20_token_price_real_time'''
    #sql_cmd_2 = '''
    #SELECT * FROM erc20_token_price_real_time WHERE datetime >= %s'''
    price_data = pd.read_sql(sql_cmd_1, engine)
    #price_data = pd.read_sql(sql_cmd_2, engine, params = [start_date])
    return price_data.groupby("pool_id")
 #%%
def get_abnormal_changes(group, threshold):
    dataframe = pd.DataFrame()
    columns = ['open', 'high', 'low', 'volume']
    close_pct_change = group['close'].pct_change().fillna(0)
    group['percent_change_C'] = close_pct_change
    group = group[group['percent_change_C'] >= threshold]
    dataframe[['datetime', 'percent_change_C','name','symbol', 'network', 'pool_id']] = group[['datetime','percent_change_C','name','symbol', 'network', 'pool_id']]
    dataset_tosave_columns = ['percent_change_O', 'percent_change_H','percent_change_L','percent_change_V']
    for idx, item in enumerate(columns):
        pct_change = group[item].pct_change().fillna(0)
        dataframe[dataset_tosave_columns[idx]] = pct_change
    return dataframe
#%%
def saveto_database(engine, database_tosave, dataframe):
    try:
        existing_data = pd.read_sql(f"SELECT datetime, pool_id FROM {database_tosave}", engine)
        existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
    except:
        existing_data = pd.DataFrame(columns=['datetime', 'pool_id'])
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    merged = dataframe.merge(existing_data, on=['datetime', 'pool_id'], how='left', indicator=True)
    to_save = merged[merged['_merge'] == 'left_only']
    to_save = to_save.drop(columns=['_merge'])
    if not to_save.empty:
        to_save.to_sql(database_tosave, engine, if_exists='append', index=False)
#%%
threshold = 0.10
database_tosave = "abnormal_price_changes"
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer"
engine = create_engine(sql_connector)
price_data = get_erc20_tokens_prices(engine)
dataframe = price_data.apply(get_abnormal_changes, threshold=threshold).reset_index(drop=True)
saveto_database(engine, database_tosave, dataframe)
