#%%
import requests
import pandas as pd
from sqlalchemy import create_engine
import time
from datetime import datetime, timedelta
from requests.exceptions import SSLError
#%%
# Get pool addresses of multi contract address 
def get_pool_address(multi_contract_address, api_url, network):
    headers = {
        'accept': 'application/json'
    }
    try:
        response = requests.get(f"{api_url}/networks/{network}/tokens/multi/{multi_contract_address}", headers = headers)
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
def read_contract_addresses(sql_connector, time_windows):
    mainnet = []
    mainnet_base = []
    mainnet_arbitrum = []
    engine = create_engine(sql_connector)
    sql_cmd = '''
    SELECT contract_address, blockchain, blocktime FROM erc_20_tokens
    '''
    data = pd.read_sql(sql_cmd, engine)
    data['blocktime'] = pd.to_datetime(data['blocktime'])
    for network, window in time_windows.items():
        filtered_data = data[data['blockchain'] == network]
        filtered_data = filtered_data[filtered_data['blocktime'] > datetime.utcnow() - timedelta(days = window)]
        if network == "mainnet":
            for address in filtered_data['contract_address']:
                mainnet.append(str(address))
        elif network == "mainnet_arbitrum":
            for address in filtered_data['contract_address']:
                mainnet_arbitrum.append(str(address))
        else:
            for address in filtered_data['contract_address']:
                mainnet_arbitrum.append(str(address))
    return mainnet, mainnet_base, mainnet_arbitrum
def save_to_database(dataframe, sql_connector, database_tosave):
    try:
        engine = create_engine(sql_connector)
        try:
            existing_pools = pd.read_sql(f"SELECT pool_id FROM {database_tosave}", engine)
        except:
            existing_pools = pd.DataFrame(columns=['pool_id'])
        dataframe = dataframe[~dataframe['pool_id'].isin(existing_pools['pool_id'])]
        dataframe.to_sql(name=database_tosave, con=engine, index=True, if_exists='append')
        engine.dispose()
    except Exception as e:
        print(f"Error saving to database: {e}")
def pool_info_storage(data,sql_connector, network, chunk, database_tosave):
    pool_data = []
    try:
        for _, item in enumerate(data['data']):
            print(item)
            top_pools_data = item['relationships']['top_pools']['data']
            if top_pools_data:
                pool_id = item['relationships']['top_pools']['data'][0]['id']
                name = item['attributes']['name']
                symbol = item['attributes']['symbol']
                coingecko_coin_id = item['attributes']['coingecko_coin_id']
                total_supply = item['attributes']['total_supply']
                price_usd = item['attributes']['price_usd']
                fdv_usd = item['attributes']['fdv_usd']
                decimals = item['attributes']['decimals']
                total_reserve_in_usd = item['attributes']['total_reserve_in_usd']
                market_cap_usd = item['attributes']['market_cap_usd']
                volume_usd_24h = item['attributes']['volume_usd']['h24']
                contract_address = item['attributes']['address']
                pool_data.append({
                    'pool_id': pool_id,
                    'name': name,
                    'symbol': symbol,
                    'total_supply': total_supply,
                    'price_usd': price_usd,
                    'fdv_usd': fdv_usd,
                    'decimals': decimals,
                    'total_reserve_in_usd': total_reserve_in_usd,
                    'market_cap_usd': market_cap_usd,
                    'volume_usd_24h': volume_usd_24h,
                    'coingecko_coin_id':coingecko_coin_id,
                    'network':network,
                    'contract_address':contract_address
                })
            else:
                print("Pool not existed!")
    except:
        print("Data error!")
    dataframe = pd.DataFrame(pool_data)
    if len(dataframe) > 0:
        save_to_database(dataframe, sql_connector,database_tosave)
def process_address(addresses, chunk_size = 5):
    chunk_addresses = []
    for j in range(0, len(addresses), chunk_size):
        chunk = addresses[j:j+chunk_size]
        chunk = ",".join(chunk)
        chunk_addresses.append(chunk)
    return chunk_addresses
def addresses_for_used(sql_connector, time_windows):
    eth_addresses, arbitrum_addresses, base_addresses = read_contract_addresses(sql_connector, time_windows)
    eth_chunk_addresses = process_address(eth_addresses)
    arbitrum_chunk_addresses = process_address(arbitrum_addresses)
    base_chunk_addresses = process_address(base_addresses)
    addresses = [eth_chunk_addresses, arbitrum_chunk_addresses, base_chunk_addresses]
    return addresses
def worker(network, multi_contract_addresses, api_url, sql_connector, database_tosave):
    if multi_contract_addresses:
        for chunk in multi_contract_addresses:
            pool_info = get_pool_address(chunk, api_url, network)
            time.sleep(2)
            if pool_info:
                pool_info_storage(pool_info, sql_connector, network, chunk, database_tosave)
#%%
sql_connector = "your_database"    
networks = ["eth","arbitrum","base"]
api_url = "https://api.geckoterminal.com/api/v2"
database_tosave = "token_pool_addresses"
time_windows = {
    'mainnet': 10,
    'mainnet_arbitrum': 35,
    'mainnet_base': 35
}
addresses = addresses_for_used(sql_connector, time_windows)
for idx, item in enumerate(addresses):
    network = networks[idx]
    worker(network, item, api_url, sql_connector, database_tosave)
# %%
