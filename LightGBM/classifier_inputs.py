#%%
import pickle
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from get_temperature import temperature
#%%
def combined_label_data(labeled_data, unlabeled_data):
    merged_dict_1 = {key: [] for key in labeled_data[0].keys()}
    for d in labeled_data:
        for key, value in d.items():
            merged_dict_1[key].append(value)
    for key, value in merged_dict_1.items():
        merged_dict_1[key] = [x for sublist in value for x in sublist]
    merged_dict_2 = {key: [] for key in unlabeled_data[0].keys() if key in ['sentence_id', 'likes', 'replies','retweets','views','date']}
    for d in unlabeled_data:
        for key, value in d.items():
            if key in ['sentence_id', 'likes', 'replies','retweets','views','date']:
                merged_dict_2[key].append(value)
    data1 = pd.DataFrame(merged_dict_1)
    data1 = data1[data1['polarity'] != 3]
    data2 = pd.DataFrame(merged_dict_2)
    data2 = data2.drop_duplicates(subset = ['sentence_id'])
    df = pd.merge(data1, data2, left_on='sentence_id', right_on='sentence_id', how = 'inner')
    df['date'] = pd.to_datetime(df['date'])
    return df
def get_market_tweets_data(data,sql_connector):
    terms = data['term'].unique()
    terms_str = ', '.join(["'" + str(term).lower() + "'" for term in terms])
    sql = f'''
    SELECT datetime, name, symbol, pool_id, close
    FROM erc20_token_price_real_time
    WHERE LOWER(symbol) IN ({terms_str}) OR LOWER(name) IN ({terms_str})
    '''
    engine = create_engine(sql_connector)
    market_df = pd.read_sql_query(sql, engine)
    market_df['symbol'] = market_df['symbol'].str.lower()
    market_df['name'] = market_df['name'].str.lower()
    market_df['date'] = pd.to_datetime(market_df['datetime'])
    data['date'] = pd.to_datetime(data['date'])
    
    # 然后再进行asof合并
    merged_slug = pd.merge_asof(market_df.sort_values('date'),
                                data.sort_values('date'), 
                                left_on='date', 
                                right_on='date', 
                                left_by='name', 
                                right_by='term', 
                                direction='backward')
    
    merged_symbol = pd.merge_asof(market_df.sort_values('date'),
                                  data.sort_values('date'), 
                                  left_on='date', 
                                  right_on='date', 
                                  left_by='symbol', 
                                  right_by='term', 
                                  direction='backward')
    merged_data = pd.concat([merged_slug, merged_symbol]).drop_duplicates().sort_values('date')
    merged_data = merged_data.drop_duplicates(subset = ['date','close','name','symbol','pool_id'])
    return merged_data
def preprocessing_temperature_polarity(merged_data):
    temperature = merged_data.groupby(['symbol','name','date','pool_id'])['temperature'].sum().reset_index()
    merged_data['weighted_polarity'] = merged_data['polarity'] * merged_data['temperature']
    weighted_polarity = merged_data.groupby(['symbol','name','date','pool_id'])['weighted_polarity'].sum().reset_index()
    weighted_polarity = weighted_polarity['weighted_polarity'] / temperature['temperature']
    prices = merged_data.groupby(['symbol','name','date','pool_id'])['close'].last().reset_index()
    result = pd.DataFrame({
    'date': temperature['date'],
    'name': temperature['name'],
    'pool_id': temperature['pool_id'],
    'symbol': temperature['symbol'],
    'weighted_polarity': weighted_polarity.values,
    'temperature': temperature['temperature'].values,
    'close': prices['close'].values
    }).reset_index(drop=True)
    return result
#%%
class data_for_lightgbm_method_2():
    def __init__(self, data, windows,multi_type,sql_connector):
        self.data = self.get_abnormal_price_changes(data,sql_connector)
        self.data['hours_from_start'] = self.data.groupby('pool_id')['date'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
        self.data = self.final_data(self.data, windows)
        self.save_df(self.data, windows,multi_type)
    def get_abnormal_price_changes(self,data, sql_connector):
        engine = create_engine(sql_connector)
        sql_cmd = '''SELECT datetime, pool_id FROM abnormal_price_changes'''
        abnormal_price_changes = pd.read_sql(sql_cmd, engine)
        merged_data = pd.merge(data, abnormal_price_changes, left_on=['date', 'pool_id'], right_on=['datetime', 'pool_id'], how='left', indicator=True)
        merged_data['_merge'] = merged_data['_merge'].apply(lambda x: 1 if x == 'both' else 0)
        merged_data.rename(columns={'_merge': 'is_abnormal_price'}, inplace=True)
        return merged_data
    def compute_temperature_average(self,data, column, hours):
        end_time = data['date'].max()
        start_time = end_time - pd.Timedelta(hours=hours)
        subset = data[(data['date'] >= start_time) & (data['date'] <= end_time)]
        end_time = data['date'].max()
        average = subset[column].mean()
        return average

    def compute_weighted_weighted_polarity_average(self,data, value_column, weight_column, hours):
        end_time = data['date'].max()
        start_time = end_time - pd.Timedelta(hours=hours)
        subset = data[(data['date'] >= start_time) & (data['date'] <= end_time)]
        total_temperature = subset[weight_column].sum()
        if total_temperature == 0:
            return np.nan
        else:
            weighted_avg = (subset[value_column] * subset[weight_column] / total_temperature).sum()
            return weighted_avg
    def final_data(self,dataset,windows):
        result = []
        dataset = dataset.reset_index(drop = True)
        grouped = dataset.groupby(['pool_id'])
        for pool_id, group in grouped:
            for date in group['date']:
                is_abnormal_price = group[group['date'] == date]['is_abnormal_price'].iloc[0]
                if is_abnormal_price == 1:
                    current_data = group[group['date'] <= date]
                    close = group[group['date'] == date]['close'].iloc[0]
                    name = group[group['date'] == date]['name'].iloc[0]
                    symbol = group[group['date'] == date]['symbol'].iloc[0]
                    hours_from_start = group[group['date'] == date]['hours_from_start'].iloc[0]
                    avg_temperatures = {
                        f'T_{hours}h': self.compute_temperature_average(current_data, 'temperature', hours)
                        for hours in windows}
                    avg_polarities = {
                        f'WWP_{hours}h': self.compute_weighted_weighted_polarity_average(current_data, 'weighted_polarity','temperature', hours)
                        for hours in windows}
                    temp_df = {
                        'date': date,
                        'pool_id': pool_id,
                        'name': name,
                        'symbol': symbol,  
                        **avg_temperatures,
                        **avg_polarities,
                        'price': close,
                        'hours_from_start': hours_from_start
                    }
                    result.append(temp_df)
        return pd.DataFrame(result)
    def save_df(self, dataset, windows, multi_type):
        if not os.path.exists(f'./dataset/en/{multi_type}/10%'):
            os.makedirs(f'./dataset/en/{multi_type}/10%')
        with open(f'./dataset/en/{multi_type}/10%/{multi_type}_{max(windows)/24}days_10%.pickle', 'wb') as file:
            pickle.dump(dataset, file) 
#%%
with open('../dataset/unlabeled/en.pkl', 'rb') as f:
    unlabeled = pickle.load(f)
with open('../dataset/labeled/en.pkl', 'rb') as f:
    labeled = pickle.load(f)
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer"
data = combined_label_data(labeled, unlabeled)
data = temperature(data)
merged_data = get_market_tweets_data(data, sql_connector)
dataset = preprocessing_temperature_polarity(merged_data)
multi_type = "multi"
windows_for_test = [[k for k in range(int(i/6), i+1, int(i/6))] for i in range(120, 1441, 12)]
for window in windows_for_test:
    data_for_lightgbm_method_2(dataset, window, multi_type,sql_connector)                                                                