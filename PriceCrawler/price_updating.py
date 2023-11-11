#%%
import requests
import pandas as pd
from sqlalchemy import create_engine
import time
import re
from datetime import datetime,timedelta
import copy
from requests.exceptions import SSLError
def read_pool_address(sql_connector, pool_database):
    engine = create_engine(sql_connector)
    sql_cmd = f'''
    SELECT pool_id, name, symbol, network FROM {pool_database}'''
    data = pd.read_sql(sql_cmd, engine)
    pool_ids = data['pool_id'].tolist()
    names = data['name'].tolist()
    symbols = data['symbol'].tolist()
    network = data['network'].tolist()
    return pool_ids, names, symbols, network
def get_ohlcv_by_pool_address(api_url, network, pool_id):
    headers = {
        'accpet':'application/json'
    }
    try:
        response = requests.get(f"{api_url}/networks/{network}/pools/{pool_id}/ohlcv/hour", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code}")
            return response.status_code
    except SSLError:
        time.sleep(2)
        print("An SSL error occurred. Please check your connection and try again.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
def remove_before_underscore(s):
    result = re.sub(r'.*?_', '', s)
    return result
def del_outdated_data(pool_id, sql_connector,database_tosave, pool_database, erc20_database):
    try:
        engine = create_engine(sql_connector)
        fetch_query = f"SELECT contract_address FROM {pool_database} WHERE pool_id = %s"
        result = engine.execute(fetch_query, (pool_id,)).fetchone()
        
        if result:
            contract_address = result[0]
            delete_from_erc20_query = f"DELETE FROM {erc20_database} WHERE contract_address = %s"
            engine.execute(delete_from_erc20_query, (contract_address,))
            print(f"Deleted data for contract_address: {contract_address} from {erc20_database}")
            delete_from_token_info_query = f"DELETE FROM {pool_database} WHERE pool_id = %s"
            engine.execute(delete_from_token_info_query, (pool_id,))
            print(f"Deleted data for pool_id: {pool_id} from {pool_database}")
            delete_from_price_database = f"DELETE FROM {database_tosave} WHERE pool_id = %s"
            engine.execute(delete_from_price_database, (pool_id,))
            print(f"Deleted data for pool_id: {pool_id} from {database_tosave}")
        else:
            print(f"No data found for pool_id: {pool_id} in {pool_database}")        
        engine.dispose()
    except Exception as e:
        print(f"Error processing database operations: {e}")
def save_to_database(dataframe, sql_connector, database_tosave):
    try:
        engine = create_engine(sql_connector)
        existing_data = pd.read_sql(f"SELECT datetime, pool_id FROM {database_tosave}", engine)
    except:
        existing_data = pd.DataFrame(columns=['datetime', 'pool_id'])
    try:
        merged = pd.merge(dataframe, existing_data, on=['datetime', 'pool_id'], how='outer', indicator=True)
        to_insert = merged[merged['_merge'] == 'left_only'][dataframe.columns]
        to_update = merged[merged['_merge'] == 'both']
        
        for index, row in to_update.iterrows():
            existing_row = existing_data[
                (existing_data['datetime'] == row['datetime']) & 
                (existing_data['pool_id'] == row['pool_id'])
            ].iloc[0]
            
            if not existing_row.equals(row):
                # 更新数据库的相应记录
                update_query = f"""
                UPDATE {database_tosave} SET
                open = %s,  
                high = %s,
                low = %s,
                close = %s,
                volume = %s
                WHERE datetime = %s AND pool_id = %s
                """
                engine.execute(update_query, (row['open'], row['high'], row['low'], row['close'], row['volume'], row['datetime'], row['pool_id']))
        
        # 插入新行
        to_insert.to_sql(name=f"{database_tosave}", con=engine, index=False, if_exists='append')
        engine.dispose()
    except Exception as e:
        print(f"Error saving to database: {e}")
def ohlcv_info_storage(ohlcv_info, sql_connector, name, symbol, network, pool_id, database_tosave, pool_database, erc20_database):
    ohlcv_info = ohlcv_info['data']['attributes']['ohlcv_list']
    if ohlcv_info:
        try:
            data_copy = copy.deepcopy(ohlcv_info)
            for row in data_copy:
                if isinstance(row[0], int):  
                    row[0] = datetime.utcfromtimestamp(row[0]).strftime('%Y-%m-%d %H:%M:%S')
                row.extend([name, symbol, network, pool_id])
            df = pd.DataFrame(data_copy, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'name', 'symbol', 'network', 'pool_id'])
            if len(df) > 0:
                latest_datetime = datetime.strptime(df.iloc[0]['datetime'], '%Y-%m-%d %H:%M:%S')
                print(df.iloc[0]['datetime'])
                print(df.iloc[-1]['datetime'])
                current_utc_datetime = datetime.utcnow()
                # 如果获取到的最新价格时间与当前差30天以上则删除该币种
                difference = current_utc_datetime - latest_datetime
                if difference <= timedelta(days=30):
                    save_to_database(df, sql_connector, database_tosave)
                else:
                    del_outdated_data(pool_id, sql_connector, database_tosave, pool_database, erc20_database)
        except:
            pass
#%%
sql_connector = "your_database"
api_url = "https://api.geckoterminal.com/api/v2"
database_tosave = "erc20_token_price_real_time"
pool_database = "token_pool_addresses"
erc20_database = "erc_20_tokens"
#%%
pool_ids, names, symbols, networks = read_pool_address(sql_connector,pool_database)
for pool_id, name, symbol,network in zip(pool_ids, names, symbols, networks):
    ohlcv_info = get_ohlcv_by_pool_address(api_url, network,remove_before_underscore(pool_id))
    if ohlcv_info:
        time.sleep(2)
        ohlcv_info_storage(ohlcv_info, sql_connector, name, symbol, network, pool_id,database_tosave, pool_database, erc20_database)
                        