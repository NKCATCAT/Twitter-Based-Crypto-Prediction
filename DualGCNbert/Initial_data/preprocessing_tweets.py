#%%
from sqlalchemy import create_engine
import pandas as pd
import os
import pickle
import json
import re
import html
from langdetect import detect
from torch.utils.data import Dataset
from multiprocessing import Pool
with open("freqwordslist.json", 'r') as f:
    freqwordslist = json.load(f)
freqwordslist = freqwordslist + ['pls', 'pc','refund','cyber', 'forge', 'pov', 'pet', 'dice', 'ate','patrick']
#%%
# 序列化问题：在并行处理的上下文中，避免使用self（即类实例的引用）是关键的。将match_tokens作为独立的函数，而不是类的方法，可以避免这个问题。
# 在实际应用中，尝试序列化与系统资源、网络连接或其他外部状态关联的对象可能会导致类似的问题。为了避免这种情况，最佳做法是将并行逻辑与这些外部状态分离，或使用不需要序列化的并发机制（如线程）
def match_terms(text, patterns_compiled):
    found_tokens = [token for pattern, token in patterns_compiled if pattern.search(text)]
    return found_tokens if found_tokens else []
class TweetsDataset(Dataset):
    def __init__(self, engine, languages):
        # Get coins
        query = """
        SELECT name, symbol, pool_id
        FROM abnormal_price_changes
        GROUP BY pool_id
        """
        coins = pd.read_sql_query(query, engine)
        coins_1 = [coin.lower() for coin in coins['name'].tolist() if coin.lower() not in freqwordslist]
        coins_2 = [coin.lower() for coin in coins['symbol'].tolist() if coin.lower() not in freqwordslist]
        new_coins_list = list(set(coins_1 + coins_2))
        del coins
        # Generate patterns for all coins
        patterns_tokens = [(pattern, coin) for coin in new_coins_list for pattern in self.generate_patterns_for_token(coin)]
        patterns_compiled = [(re.compile(pattern), token) for pattern, token in patterns_tokens]
        # Fetch and clean tweets
        tweets_df = pd.read_sql_query("SELECT * FROM tweets_df_realtime_update", engine)
        tweets_df['clean_text'] = tweets_df['text'].map(self.clean)
        tweets_df['clean_text_list'] = tweets_df['clean_text'].apply(lambda x: x.split(" "))
        # Find matched terms
        matched_terms_list = tweets_df['clean_text'].apply(lambda x: match_terms(x, patterns_compiled)).tolist()
        mask = [len(term_list) > 0 for term_list in matched_terms_list]
        relevant_df_chunk = tweets_df[mask].copy()
        relevant_df_chunk['matched_terms'] = [matched_terms_list[i] for i, m in enumerate(mask) if m]
 
        if len(relevant_df_chunk) > 0:
            relevant_df_chunk['language'] = relevant_df_chunk['clean_text'].apply(self.detect_language)
            if languages == 'en':
                mask = relevant_df_chunk['language'] == 'en'
                relevant_df_chunk = relevant_df_chunk[mask].copy()

            relevant_df_chunk['term_positions'] = relevant_df_chunk.apply(self.find_term_positions, axis = 1)
            relevant_df_chunk['term_and_term_positions'] = relevant_df_chunk.apply(lambda row: list(zip(row['matched_terms'], row['term_positions'])), axis=1)
            relevant_df_chunk = relevant_df_chunk.drop(['matched_terms', 'term_positions'], axis=1)
            relevant_df_chunk['sentence_id'] = range(1, len(relevant_df_chunk) + 1)
            relevant_df_chunk = relevant_df_chunk.explode('term_and_term_positions')
            relevant_df_chunk[['term', 'term_positions']] = pd.DataFrame(relevant_df_chunk['term_and_term_positions'].tolist(), index = relevant_df_chunk.index)
            relevant_df_chunk['group'] = relevant_df_chunk['sentence_id'].astype(str) + "_" + relevant_df_chunk['term']
            relevant_df_chunk['term_id'] = (relevant_df_chunk.groupby('group').ngroup() + 1).astype(str)
            relevant_df_chunk = relevant_df_chunk.drop(['term_and_term_positions','group'], axis=1) 
        self.relevant_df = relevant_df_chunk
        self.save_df()
    def __len__(self):
        return len(self.relevant_df)

    def __getitem__(self, idx):
        return self.relevant_df.iloc[idx]   
    def generate_patterns_for_token(self, token):
        patterns = [
            r'\b' + re.escape(token) + r'\b',
            r'\$\b' + re.escape(token) + r'\b',
            r'\$\b' + re.escape(token) + r'\$\b',
            r'\b' + re.escape(token) + r'\$\b',
            r'￥\b' + re.escape(token) + r'\b',
            r'￥\b' + re.escape(token) + r'￥\b',
            r'\b' + re.escape(token) + r'￥\b'
        ]
        return patterns
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
    def find_term_positions(self, row):
        term_tokens = row['matched_terms']
        positions = []
        
        # Convert the tokenized text back to a string for regex matching
        clean_text_str = row['clean_text']

        for token in term_tokens:
            patterns = self.generate_patterns_for_token(token)
            for pattern in patterns:
                for match in re.finditer(pattern, clean_text_str):
                    start_idx = len(clean_text_str[:match.start()].split())  # Count number of words before match
                    end_idx = start_idx + len(match.group().split())  # Add number of words in match
                    positions.append((str(start_idx), str(end_idx)))  # end_idx - 1 since it's inclusive
        return positions

    def df_to_list(self):
        result = []
        for _, row in self.relevant_df.iterrows():
            aspects = []
            aspect = {
                    'term': [row['term']],
                    'from': row['term_positions'][0], 
                    'to': row['term_positions'][1],
                    'polarity': None
                }
            aspects.append(aspect)
            result.append({
                'aspects': aspects,
                'token': row['clean_text_list'],
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
        if not os.path.exists('../dataset/unlabeled'):
            os.makedirs('../dataset/unlabeled')
        with open('../dataset/unlabeled/en.pkl', 'w') as file:
            pickle.dump(dataset, file)
#%%
engine = create_engine("mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer")
dataset = TweetsDataset(engine, languages='en')
engine.dispose()