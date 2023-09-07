# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:40:42 2023

@author: 86189
"""

import pickle
import os
import json
from datetime import timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from get_temperature import temperature
with open('../DualGCNbert/dataset/unlabeled/en/en.json', 'r') as f:
    unlabeled = json.load(f)
with open('../DualGCNbert/dataset/labeled/en/en.pkl', 'rb') as f:
    labeled = pickle.load(f)
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
    data['term'] = data['term'].replace("$", "")
    terms = data['term'].unique()
    terms_str = ', '.join(["'" + str(term).lower() + "'" for term in terms])
    sql = f'''
    SELECT time, prices, slug, symbol
    FROM market_hourly_data_final
    WHERE LOWER(slug) IN ({terms_str}) OR LOWER(symbol) IN ({terms_str})
    '''
    engine = create_engine(sql_connector)
    market_df = pd.read_sql_query(sql, engine)
    market_df['symbol'] = market_df['symbol'].str.lower()
    market_df['time'] = pd.to_datetime(market_df['time'])
    data['date'] = pd.to_datetime(data['date'])
    
    # 然后再进行asof合并
    merged_slug = pd.merge_asof(market_df.sort_values('time'),
                                data.sort_values('date'), 
                                left_on='time', 
                                right_on='date', 
                                left_by='slug', 
                                right_by='term', 
                                direction='backward')
    
    merged_symbol = pd.merge_asof(market_df.sort_values('time'),
                                  data.sort_values('date'), 
                                  left_on='time', 
                                  right_on='date', 
                                  left_by='symbol', 
                                  right_by='term', 
                                  direction='backward')
    merged_data = pd.concat([merged_slug, merged_symbol]).drop_duplicates().sort_values('time')
    merged_data = merged_data.drop_duplicates(subset = ['time','prices','slug','symbol'])
    return merged_data
def preprocessing_temperature_polarity(merged_data):
    temperature = merged_data.groupby(['symbol','slug','time'])['temperature'].sum().reset_index()
    merged_data['weighted_polarity'] = merged_data['polarity'] * merged_data['temperature']
    weighted_polarity = merged_data.groupby(['symbol','slug','time'])['weighted_polarity'].sum().reset_index()
    weighted_polarity = weighted_polarity['weighted_polarity'] / temperature['temperature']
    prices = merged_data.groupby(['symbol','slug','time'])['prices'].last().reset_index()
    result = pd.DataFrame({
    'date': temperature['time'],
    'symbol': temperature['symbol'],
    'weighted_polarity': weighted_polarity.values,
    'temperature': temperature['temperature'].values,
    'price': prices['prices'].values
    }).reset_index(drop=True)
    return result
#%%
class data_for_lightgbm_method_1():
    def __init__(self, dataset):
        data = self.final_data(dataset)
        self.data = pd.DataFrame(data)
        self.save_df(data)
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
    
        # Only consider rows where neither value_column nor weight_column is NaN
        valid_rows = subset.dropna(subset=[value_column, weight_column])
    
        total_temperature = valid_rows[weight_column].sum()
    
        # If total_temperature is 0, then the weighted average is NaN
        if total_temperature == 0:
            return np.nan
    
        weighted_avg = (valid_rows[value_column] * valid_rows[weight_column]).sum() / total_temperature
    
        return weighted_avg
    def compute_price_pctchange(self,data):
        end_price = data['price'].iloc[-1]
        target_time = data['date'].iloc[-1] + pd.Timedelta(hours = 120)
        closest_row = data.iloc[(data['date'] - target_time).abs().idxmin()]
        start_price = closest_row['price']
        pct_change = (end_price - start_price) / start_price * 100
        if pct_change >= 100:
            pct_change = 1
        else:
            pct_change = 0
        return pct_change
    def final_data(self,dataset):
        result = []
        grouped = dataset.groupby('term')
        for name, group in grouped:
            for date in group['date']:
                current_data = group[group['date'] <= date]
                is_double = self.compute_price_pctchange(current_data)
                avg_temperatures = {
                    f'T_{hours}h': self.compute_temperature_average(current_data, 'temperature', hours)
                    for hours in [6, 12, 24, 48, 96, 168, 240]}
                avg_polarities = {
                    f'WWP_{hours}h': self.compute_weighted_weighted_polarity_average(current_data, 'weighted_polarity','temperature', hours)
                    for hours in [6, 12, 24, 48, 96, 168, 240]}
                hours_from_start = (date - group['date'].min()).dt.total_seconds / 3600
                temp_df = {
                    'date': date,
                    'coin': name,
                    'hours': hours_from_start,
                    **avg_temperatures,
                    **avg_polarities,
                    'is_double':is_double
                    }
                result.append(temp_df)
        return result
    def save_df(self, dataset):
        if not os.path.exists('../dataset/lightgbm/en'):
            os.makedirs('../dataset/lightgbm/en')
        with open('../dataset/lightgbm/en/method1.pickle', 'wb') as file:
            pickle.dump(dataset, file)
#%%
class data_for_lightgbm_method_2():
    def __init__(self, dataset, windows,abnormal_threshold,multi_type):
        data = self.abnormal_price_change(dataset, windows,abnormal_threshold)
        self.data = self.final_data(data, windows)
        self.save_df(self.data, windows,multi_type)
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
        grouped = dataset.groupby(['symbol'])
        for name, group in grouped:
            for date in group['date']:
                is_abnormal_price = group[group['date'] == date]['is_abnormal_change'].iloc[0]
                if is_abnormal_price == 1:
                    current_data = group[group['date'] <= date]
                    price = group[group['date'] == date]['price'].iloc[0]
                    hours_from_start = group[group['date'] == date]['hours_from_start'].iloc[0]
                    is_double = group[group['date'] == date]['is_double'].iloc[0]
                    growth_rate = group[group['date'] == date]['double_ratio'].iloc[0]
                    double_date = group[group['date'] == date]['max_doubled_date'].iloc[0]
                    double_price = group[group['date'] == date]['max_doubled_in_future'].iloc[0]
                    avg_temperatures = {
                        f'T_{hours}h': self.compute_temperature_average(current_data, 'temperature', hours)
                        for hours in windows}
                    avg_polarities = {
                        f'WWP_{hours}h': self.compute_weighted_weighted_polarity_average(current_data, 'weighted_polarity','temperature', hours)
                        for hours in windows}
                    temp_df = {
                        'date': date,
                        'coin': name,
                        'is_double': is_double,
                        'double_date':double_date,
                        **avg_temperatures,
                        **avg_polarities,
                        'price': price,
                        'double_price': double_price,
                        'growth_rate': growth_rate,
                        'hours_from_start': hours_from_start
                    }
                    result.append(temp_df)
        return pd.DataFrame(result)
    def abnormal_price_change(self,dataset,windows,abnormal_threshold):
        def optimized_check_double_price_and_date(group):
            group = group.sort_values(by = 'date')
            n = len(group)
            doubled_prices = [None] * n
            doubled_dates = [None] * n
            is_double = [0] * n
            double_ratios = [None] * n

            for idx in range(n):
                current_price = group.iloc[idx]['price']
        
                # Find future prices that are at least double the current price
                future_doubled = group.iloc[idx+1:]['price'].ge(2 * current_price)
                if future_doubled.sum() > 0:
                    # Get the max doubled price and its date from the future prices
                    max_doubled_price = group.iloc[idx+1:][future_doubled]['price'].max()
                    max_doubled_date = group.iloc[idx+1:][group['price'] == max_doubled_price]['date'].values[0]
                    ratio = (max_doubled_price - current_price) / current_price

                    doubled_prices[idx] = max_doubled_price
                    doubled_dates[idx] = max_doubled_date
                    is_double[idx] = 1
                    double_ratios[idx] = ratio

            group['max_doubled_in_future'] = doubled_prices
            group['max_doubled_date'] = doubled_dates
            group['is_double'] = is_double
            group['double_ratio'] = double_ratios
            return group
        def optimized_check_double_price_and_date_v2(group):
            group = group.sort_values(by = 'date')
            # 使用索引来获取未来的价格
            max_doubled_prices = []
            max_doubled_dates = []
            is_doubles = []
            double_ratios = []

            prices = group['price'].values
            dates = group['date'].values

            for idx, price in enumerate(prices):
                # 取得未来的价格
                future_prices = prices[idx+1:]

                # 找到大于或等于两倍当前价格的价格
                doubled_prices = future_prices[future_prices >= 2 * price]

                if len(doubled_prices) > 0:
                    max_doubled_price = doubled_prices.max()
                    max_doubled_date = dates[idx+1:][(prices[idx+1:] == max_doubled_price).argmax()]
                    max_doubled_prices.append(max_doubled_price)
                    max_doubled_dates.append(max_doubled_date)
                    is_doubles.append(1)
                    double_ratios.append((max_doubled_price / price) - 1)
                else:
                    max_doubled_prices.append(None)
                    max_doubled_dates.append(None)
                    is_doubles.append(0)
                    double_ratios.append(None)

            group['max_doubled_in_future'] = max_doubled_prices
            group['max_doubled_date'] = max_doubled_dates
            group['is_double'] = is_doubles
            group['double_ratio'] = double_ratios

            return group

        def get_abnormal_change(group, abnormal_change):
            pctchange = group['price'].pct_change()
            pctchange[pctchange >= abnormal_change] = 1
            pctchange[pctchange <= -abnormal_change] = -1
            pctchange[pctchange.between(-abnormal_change, abnormal_change)] = 0
            group['is_abnormal_change'] = pctchange
            group = optimized_check_double_price_and_date_v2(group)
            result = group[group['date'] <= (group['date'].min() + timedelta(hours=max(windows)))]
            result['hours_from_start'] = (result['date'] - group['date'].min()).dt.total_seconds() / 3600
            return result
        df = dataset.sort_values(by='date').groupby('symbol').apply(get_abnormal_change,abnormal_change=abnormal_threshold)
        return df
    def save_df(self, dataset, windows, multi_type, abnormal_threshold):
        if not os.path.exists(f'./dataset/en/{multi_type}/{abnormal_threshold}'):
            os.makedirs('./dataset/en/{multi_type}/{abnormal_threshold}')
        with open(f'./dataset/en/{multi_type}/{abnormal_threshold}/{multi_type}_{max(windows)/24}days_{abnormal_threshold*100}%.pickle', 'wb') as file:
            pickle.dump(dataset, file)
#%%
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer"
data = combined_label_data(labeled, unlabeled)
data = temperature(data)
merged_data = get_market_tweets_data(data, sql_connector)
dataset = preprocessing_temperature_polarity(merged_data)
#%%
#data_for_lightgbm_method_1(dataset)
abnormal_threshold = 0.1
multi_type = "multi" # non_multi
if multi_type == "multi":
    windows_for_test = [[k for k in range(int(i/6), i+1, int(i/6))] for i in range(120, 1441, 12)]
else:
    windows_for_test = [[i] for i in range(120, 1441, 12)]
for window in windows_for_test:
    data_for_lightgbm_method_2(dataset, window,abnormal_threshold, multi_type)
