#%%
import re
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from multiprocessing import Pool

def generate_patterns_for_token(token):
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

def find_matches_for_tweet_subset(patterns, tweet_subset):
    combined_tweets = " ".join(tweet_subset)
    regex = re.compile("|".join(patterns), re.IGNORECASE)
    return set(regex.findall(combined_tweets))

def read_twitter_mentioning(sql_connector):
    now = datetime.utcnow()
    amonthago = now - timedelta(days=30)
    engine = create_engine(sql_connector)

    sql_cmd_1 = f'''SELECT text FROM tweets_df_realtime_update WHERE
    date BETWEEN '{amonthago}' AND '{now}' '''  
    sql_cmd_2 = f'''SELECT name, symbol FROM token_stats'''

    recent_tweets_df = pd.read_sql(sql_cmd_1, engine)
    tokens_df = pd.read_sql(sql_cmd_2, engine)

    recent_tweets_list = recent_tweets_df['text'].tolist()

    all_tokens = tokens_df['name'].tolist() + tokens_df['symbol'].tolist()
    all_patterns = [pattern for token in all_tokens for pattern in generate_patterns_for_token(token)]

    # Split tweets list for 10 processes
    chunk_size = len(recent_tweets_list) // 10
    tweet_subsets = [recent_tweets_list[i:i + chunk_size] for i in range(0, len(recent_tweets_list), chunk_size)]

    # Parallelize regex searching using 10 processes
    with Pool(10) as p:
        matches_list = p.starmap(find_matches_for_tweet_subset, [(all_patterns, subset) for subset in tweet_subsets])
    
    # Combine the results
    matches = set().union(*matches_list)
    mentioned = [(name, symbol) for name, symbol in zip(tokens_df['name'], tokens_df['symbol']) if name in matches or symbol in matches]

    engine.dispose()
    return mentioned



def update_unmentioned_days(sql_connector):
    engine = create_engine(sql_connector)
    
    # 获取token_stats表中的所有tokens和他们的未被提及的天数
    token_stats_df = pd.read_sql('SELECT name, symbol, unmentioned_days FROM token_stats', engine)

    # 获取这个月提及的所有tokens
    mentioned_tokens = read_twitter_mentioning(sql_connector)
    mentioned_df = pd.DataFrame(mentioned_tokens, columns=['name', 'symbol'])
    
    # 对于不在提及列表中的tokens，增加他们的未被提及的天数
    merge_df = pd.merge(token_stats_df, mentioned_df, on=['name', 'symbol'], how='left', indicator=True)
    merge_df['unmentioned_days'] = merge_df['unmentioned_days'].astype(int)
    merge_df.loc[merge_df['_merge'] == 'left_only', 'unmentioned_days'] += 1
    # 删除连续30天未被提及的tokens
    to_delete = merge_df[merge_df['unmentioned_days'] > 30]
    for _, row in to_delete.iterrows():
        engine.execute('DELETE FROM erc_20_tokens WHERE name = ? AND symbol = ?',row['name'], row['symbol'])
        engine.execute('DELETE FROM token_stats WHERE name = ? AND symbol = ?', row['name'], row['symbol'])
    # 更新未被提及天数
    to_update = merge_df[merge_df['_merge'] == 'left_only']
    for _, row in to_update.iterrows():
        engine.execute('UPDATE token_stats SET unmentioned_days = ? WHERE name = ? AND symbol = ?', row['unmentioned_days'], row['name'], row['symbol'])
    engine.dispose() 
def add_new_tokens_to_stats(sql_connector):
    engine = create_engine(sql_connector)
    
    # 从erc_20_tokens获取所有的tokens
    all_tokens_df = pd.read_sql('SELECT name, symbol, contract_address FROM token_pool_addresses', engine)
    # 从token_stats获取所有的tokens
    try:
        tracked_tokens_df = pd.read_sql('SELECT name, symbol, contract_address FROM token_stats', engine)
    except:
        tracked_tokens_df = pd.DataFrame(columns=['name', 'symbol', 'contract_address', 'unmentioned_days'])
        tracked_tokens_df.to_sql('token_stats', engine, index=False)
    # 找出erc_20_tokens中的新tokens
    merged_df = pd.merge(all_tokens_df, tracked_tokens_df, on=['name', 'symbol','contract_address'], how='left', indicator=True)
    new_tokens_df = merged_df[merged_df['_merge'] == 'left_only'][['name', 'symbol','contract_address']]
    new_tokens_df['unmentioned_days'] = 0

    # 将新tokens添加到token_stats
    new_tokens_df.to_sql('token_stats', engine, if_exists='append', index=False)

    engine.dispose()
#%%
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer" 
add_new_tokens_to_stats(sql_connector)
update_unmentioned_days(sql_connector)
