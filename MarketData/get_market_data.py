import requests
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sqlalchemy import create_engine
#%%
def get_coin_id_list(x_cg_pro_api_key):
    url = "https://pro-api.coingecko.com/api/v3/coins/list"
    headers = {
        "x-cg-pro-api-key": f'{x_cg_pro_api_key}'
        }
    response = requests.get(url, headers=headers)
    coin_list = response.json()
    return coin_list
def get_market_rank(filepath):
    with open(filepath, 'rb') as f:
        market_rank = pickle.load(f)
    return market_rank
def get_historical_data(coin_id, from_timestamp, to_timestamp, x_cg_pro_api_key):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    headers = {
        "x-cg-pro-api-key": f'{x_cg_pro_api_key}'}
    params = {
        "vs_currency": "usd", 
        "from": date_to_timestamp(from_timestamp), 
        "to": date_to_timestamp(to_timestamp)
    }
    response = requests.get(url, headers=headers, params = params)
    data = response.json()
    return data
def historical_data_to_df(data_dict):
    prices_df = pd.DataFrame(data_dict['prices'], columns= ['time', 'prices'])
    market_cap_df = pd.DataFrame(data_dict['market_caps'], columns = ['time', 'market_cap'])
    volumes_df = pd.DataFrame(data_dict['total_volumes'], columns = ['time', 'volumes'])
    
    df_list = [prices_df, market_cap_df, volumes_df]
    
    for df in df_list:
        df['time'] = pd.to_datetime(df['time'], unit = 'ms')
        
    for df in df_list[1:]:
        df_list[0] = pd.merge(df_list[0], df, on = 'time')
    
    return df_list[0]
def date_to_timestamp(date_string):
    dt = datetime.strptime(date_string, '%Y-%m-%d')
    return dt.timestamp()
#%%
# hourly
def get_hourly_data(x_cg_pro_api_key,market_rank_list,sql_connector, database_tosave):
    def get_hourly_data_by_coin(coin_id,x_cg_pro_api_key):
        start_date = datetime(2022, 12, 15)
        end_date = datetime(2023, 7, 24)
        
        days_per_call = 90
        
        current_date = start_date
        data = []
        while current_date <= end_date:
            call_end_date = min(current_date + timedelta(days=days_per_call), end_date)
            try:
                data_dict = get_historical_data(coin_id, current_date.strftime('%Y-%m-%d'), call_end_date.strftime('%Y-%m-%d'),x_cg_pro_api_key)
                df = historical_data_to_df(data_dict)
                data.append(df)
                current_date = call_end_date + timedelta(days=1)
            except:
                continue 
        data_df = pd.concat(data)
        return data_df
    for coin in market_rank_list:
        print(coin)
        data_hourly = get_hourly_data_by_coin(coin)
        if len(data_hourly) > 0:
            for _, row in market_rank.iterrows():
                slug = row['slug']
                cmc_rank = row['cmc_rank']
                if slug == coin:
                    data_hourly['slug'] = slug
                    data_hourly['cmc_rank'] = cmc_rank
                    data_hourly['symbol'] = market_rank['symbol'].loc[market_rank['slug'] == slug].iloc[0]
            engine = create_engine(sql_connector)
            chunk_size = 1000
            for i in range(0, len(data_hourly), chunk_size):
                data_hourly[i:i+chunk_size].to_sql(name = database_tosave, con = engine, index = True, if_exists = 'append')
            engine.dispose()
#%%
x_cg_pro_api_key = 'CG-Ryp3NHHj6bx1zo84RvG1vG9x'
filepath = r"./cmc_rank.pickle"
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer" 
database_tosave = "market_hourly_data_final"
# list
coin_list = get_coin_id_list(x_cg_pro_api_key)
market_rank = get_market_rank(filepath)
coin_list = [coin['id'] for coin in coin_list]
market_rank = market_rank[market_rank['slug'].isin(coin_list)]
market_rank.reset_index(drop = True, inplace = True)
market_rank_list = market_rank['slug'].tolist()
# get data
get_hourly_data(x_cg_pro_api_key,market_rank_list,sql_connector, database_tosave)