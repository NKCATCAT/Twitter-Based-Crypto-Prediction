# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:38:13 2023

@author: 86189
"""

from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import TweetTokenizer
import os
import json
import re
import html
from langdetect import detect
from torch.utils.data import Dataset
import jieba
with open("freqwordslist.json", 'rb') as f:
    freqwordslist = json.load(f)
freqwordslist = freqwordslist + ['pls', 'pc','refund','cyber', 'forge', 'pov', 'pet', 'dice', 'ate','patrick']
#%%
class TweetsDataset(Dataset):
    def __init__(self, engine, languages):
        # Load data from database
        query = """
        SELECT slug, symbol
        FROM (
            SELECT slug ,symbol, MIN(time) as min_time
            FROM market_hourly_data_final
            GROUP BY slug
        ) subquery
        WHERE min_time >= '2023-01-01'
        """
        coins = pd.read_sql_query(query, engine)
        coins_1 = [coin.lower() for coin in coins['slug'].tolist() if coin.lower() not in freqwordslist]
        coins_2 = [coin.lower() for coin in coins['symbol'].tolist() if coin.lower() not in freqwordslist]
        self.new_coins_list = list(set(coins_1 + coins_2))
        del coins
        
        
        tweets_df = pd.read_sql_query("SELECT * FROM tweets_data", engine)
        tweets_df['clean_text'] = tweets_df['text'].map(self.clean)
        pattern = '|'.join(self.new_coins_list)
        mask = tweets_df['clean_text'].str.contains(pattern, na = False)
        relevant_df_chunk = tweets_df[mask].copy()
        if len(relevant_df_chunk) > 0:
            relevant_df_chunk['language'] = relevant_df_chunk['clean_text'].apply(self.detect_language)
            relevant_df_chunk['term'] = relevant_df_chunk['clean_text'].apply(self.find_coins, args=(self.new_coins_list,))
            relevant_df_chunk.dropna(subset=['term'], inplace=True)  
            
            if languages == 'en':
                mask = relevant_df_chunk['language'] == 'en'
                relevant_df_chunk = relevant_df_chunk[mask].copy()
                self.tokenizer = TweetTokenizer()
                relevant_df_chunk['token'] = relevant_df_chunk['clean_text'].apply(self.tokenizer.tokenize)
            else:
                mask = relevant_df_chunk['language'].str[:2] == 'zh'
                relevant_df_chunk = relevant_df_chunk[mask].copy()
                relevant_df_chunk['token'] = relevant_df_chunk['clean_text'].apply(jieba.lcut)
                
            relevant_df_chunk['term_positions'] = relevant_df_chunk.apply(self.find_term_positions, axis = 1)
            relevant_df_chunk['term_and_positions'] = relevant_df_chunk.apply(lambda row: list(zip(row['term'], row['term_positions'])), axis=1)
            relevant_df_chunk = relevant_df_chunk.dropna(subset=['term_and_positions'])
            relevant_df_chunk = relevant_df_chunk.drop(['term', 'term_positions'], axis=1)
            relevant_df_chunk['sentence_id']= range(1, len(relevant_df_chunk) + 1)
            relevant_df_chunk = relevant_df_chunk.explode('term_and_positions')
            relevant_df_chunk[['term', 'term_positions']] = pd.DataFrame(relevant_df_chunk['term_and_positions'].tolist(), index = relevant_df_chunk.index)
            relevant_df_chunk = relevant_df_chunk.dropna(subset = ['term'])
            relevant_df_chunk['group'] = relevant_df_chunk['sentence_id'].astype(str) + "_" + relevant_df_chunk['term']
            relevant_df_chunk['term_id'] = pd.factorize(relevant_df_chunk['group'])[0] + 1
            relevant_df_chunk = relevant_df_chunk.drop(['term_and_positions'], axis=1)  

        # Concatenate all processed chunks into one DataFrame
        self.relevant_df = relevant_df_chunk
        self.save_df()
    def __len__(self):
        return len(self.relevant_df)

    def __getitem__(self, idx):
        return self.relevant_df.iloc[idx]

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return None
        
    def clean(self, text):
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text.strip()

    def find_coins(self, text, coin_list):
        coin_list = [r'\b' + coin + r'\b|\$\b' + coin + r'\b' for coin in coin_list] # 在每个币种名称前后添加 \b 边界元字符
        pattern = '|'.join(coin_list)
        match = re.findall(pattern, text)
        if match:
            return match
        return None


    def find_term_positions(self, row):
        term_tokens = row['term']
        positions = [(i, i + 1) for i, token in enumerate(row['token']) if token in term_tokens]
        return positions
    
    def df_to_list(self):
        result = []
        for _, row in self.relevant_df.iterrows():
            aspects = []
            try:
                aspect = {
                        'term': [row['term']],
                        'from': row['term_positions'][0], 
                        'to': row['term_positions'][1],
                        'polarity': None
                    }
            except:
                print(row)
            aspects.append(aspect)
            result.append({
                'aspects': aspects,
                'token': row['token'],
                'sentence_id': row['term_id'],
                'replies': row['replies'],
                'retweets': row['retweets'],
                'likes': row['likes'],
                'views': row['views'],
                'date': str(row['date']),
            })
        return result
 
    def save_df(self):
        dataset = self.df_to_list()
        if not os.path.exists('../dataset/unlabeled/en'):
            os.makedirs('../dataset/unlabeled/en')
        with open('../dataset/unlabeled/en/en.json', 'w') as file:
            json.dump(dataset, file)
            
#%%
engine = create_engine("mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer")
dataset = TweetsDataset(engine, languages='en')
engine.dispose()
