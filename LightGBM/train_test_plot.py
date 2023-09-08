# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:52:18 2023

@author: 86189
"""
import pickle
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix,auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re
import numpy as np
import statistics
from sqlalchemy import create_engine
from adjustText import adjust_text
import math
np.random.seed(42)
#%%
# lightgbm training and test and evaluation metrics
def train_and_test(train_data, val_data, x_val,y_val,coin_names_val,growth_rates, double_date, price, double_price,
                   current_date,features):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'seed':42,
        'num_threads': 1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'deterministic':True
    }
    # 训练轮数
    num_round = 100
    bst = lgb.train(params, train_data, num_round,valid_sets = [val_data])
    # 进行预测
    y_pred_prob = bst.predict(x_val)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]
    y_val = y_val.reset_index(drop = True)
    '''
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize= (16, 8), dpi = 400)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    cnf_matrix = confusion_matrix(y_val, y_pred)
    print(f"True Positives: {cnf_matrix[1][1]}")
    print(f"False Positives: {cnf_matrix[0][1]}") 
    print(f"True Negatives: {cnf_matrix[0][0]}")
    print(f"False Negatives: {cnf_matrix[1][0]}")
    '''
    # 计算评估指标
    acc = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    '''
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    '''
    
    '''
    # feature importances
    feature_importances = bst.feature_importance(importance_type='gain')
    feature_name = bst.feature_name()
    sorted_idx = feature_importances.argsort()
    plt.figure(figsize=(10, 12), dpi = 400)
    sns.barplot(x=feature_importances[sorted_idx], y=[feature_name[i] for i in sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()
    '''
    return {
    "Accuracy": acc,
    "ROC AUC": roc_auc,
    "Precision": precision,
    "Recall": recall,
    "y_pred_prob":y_pred_prob,
    "y_val":y_val,
    "y_pred":y_pred,
    "coin_names_val": coin_names_val,
    "growth_rate": list(growth_rates),
    "double_date": list(double_date),
    "double_price": list(double_price),
    "price":list(price),
    "current_date":list(current_date)
    }
#%%
def extract_number(text):
    match = re.search(r'(\d+\.\d+)', text)
    if match:
        return float(match.group(1))
    return 0
#%%
def get_lightgbm_results(filepath, features_type, testing_pct,threshold):
    results = []
    folder = os.listdir(filepath+features_type+threshold)
    folder = sorted(folder,key = extract_number)
    for datapath in folder:
        with open(datapath, 'rb') as f:
            tweets = pickle.load(f)
        match = re.search(r'(\d+\.\d+)days', datapath)
        if match:
            number = int(float(match.group(1)) * 24)
        tweets = tweets[tweets[f'T_{number}h'] > 0] 
        tweets['coin'] = tweets['coin'].apply(lambda x: x[0])
        all_columns = set(tweets.columns)
        exclude_columns = set(['is_double', 'date', 'coin', 'double_date','price','growth_rate','double_date','double_price'])
        features = list(all_columns - exclude_columns)
        target = ['is_double']
        unique_coins = tweets['coin'].unique()
        half_length = int(testing_pct * len(unique_coins))
        val_coins = unique_coins[:half_length]
        # 使用这些币种筛选出验证集
        val_mask = tweets['coin'].isin(val_coins)
        X_val = tweets[val_mask][features]
        y_val = tweets[val_mask][target]
        coin_names_val = tweets[val_mask]['coin']
        growth_rates = tweets[val_mask]['growth_rate']
        price = tweets[val_mask]['price']
        double_date = tweets[val_mask]['double_date']
        double_price = tweets[val_mask]['double_price']
        current_date = tweets[val_mask]['date']
        
        # 剩下的数据是训练集
        X_train = tweets[~val_mask][features]
        y_train = tweets[~val_mask][target]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        metrics = train_and_test(train_data, val_data, X_val, y_val, coin_names_val,growth_rates,double_date, price, double_price,current_date,features)
        results.append(metrics)
    return results, unique_coins
#%%
# 功能一：看不同策略的胜率
def win_rate_by_coin(y_pred_prob, y_val, y_pred, coin_names_val,growth_rate,double_date, price, double_price,current_date, n):
    unique_coins = set(coin_names_val)
    total_positions = 0
    wins = 0    
    strategy_growth_rates = []
    strategy_double_dates = []
    strategy_prices = []
    strategy_double_prices = []
    strategy_current_dates = []
    strategy_coins = []
    for coin in unique_coins:
        coin_indices = [i for i, coin_id in enumerate(coin_names_val) if coin_id == coin]
        coin_probs = [(y_pred_prob[i], y_val.iloc[i], y_pred[i], growth_rate[i], double_date[i],
                       price[i], double_price[i], current_date[i]) for i in coin_indices]
        double_predictions = [p for p in coin_probs if p[2] == 1]
        if len(double_predictions) >= n:
            _, actual, pred, gr,dd,p,dp,cd = double_predictions[n-1]
            total_positions += 1
            actual = int(actual.values)
            if actual == pred and pred == 1:
                wins += 1
                strategy_coins.append(coin)
                strategy_current_dates.append(cd)
                strategy_double_dates.append(dd)
                strategy_double_prices.append(dp)
                strategy_growth_rates.append(gr)
                strategy_prices.append(p)

    median_growth_rate = statistics.median(strategy_growth_rates) if strategy_growth_rates else 0
    win_rate = wins / total_positions if total_positions > 0 else 0

    strategy_dataframe = pd.DataFrame({
    'current_date': strategy_current_dates,
    'prices': strategy_prices,
    'coins': strategy_coins,
    'double_prices': strategy_double_prices,
    'double_dates': strategy_double_dates,
    'growth_rates': strategy_growth_rates
    })
    return win_rate, wins, total_positions, median_growth_rate, strategy_dataframe
#%%
# 功能二：记录所有预测
def get_model_predictions_records(results):
    predictions_records = []
    def predictions_record(y_pred_prob, y_val, y_pred, coin_names_val,growth_rate,double_date, price, double_price,current_date):
        unique_coins = set(coin_names_val)
        successful_predictions = []
        failed_predictions = []
        
        for coin in unique_coins:
            coin_indices = [i for i, coin_id in enumerate(coin_names_val) if coin_id == coin]
            coin_probs = [(y_pred_prob[i], y_val.iloc[i], y_pred[i], growth_rate[i], double_date[i],
                           price[i], double_price[i], current_date[i]) for i in coin_indices]
            double_predictions = [p for p in coin_probs if p[2] == 1]
            if len(double_predictions) > 0:
                for _, actual, pred, gr, dd, p, dp, cd in double_predictions:
                    actual = int(actual.values)
                    if pred == 1:
                        if pred == actual:
                            successful_predictions.append((cd, p, coin, dp, dd, gr))
                        else:
                            failed_predictions.append((cd, p, coin, dp, dd, gr))
        success_df = pd.DataFrame(successful_predictions, columns=['current_date', 'prices', 'coins', 
                                                                  'double_prices', 'double_dates', 'growth_rates'])
        success_df['status'] = 'Success'
    
        fail_df = pd.DataFrame(failed_predictions, columns=['current_date', 'prices', 'coins', 
                                                            'double_prices', 'double_dates', 'growth_rates'])
        fail_df['status'] = 'Fail'
    
        dataframe = pd.concat([success_df, fail_df], axis=0).drop_duplicates()
        return dataframe
    for item in results:
        dataframe = predictions_record(item['y_pred_prob'], item['y_val'], 
                                                           item['y_pred'],item['coin_names_val'],item['growth_rate'],
                                                           item['double_date'],item['price'],item['double_price'],
                                                           item['current_date'])
        predictions_records.append(dataframe)
    return predictions_records
# gather win rates from different strategies.
def get_win_rate(data, n):
    win_rate_data = []
    dataframes = []
    for item in data:
        win_rate, wins, total_positions, avg_growth_rate,strategy_dataframe = win_rate_by_coin(item['y_pred_prob'], item['y_val'], 
                                                           item['y_pred'],item['coin_names_val'],item['growth_rate'],
                                                           item['double_date'],item['price'],item['double_price'],
                                                           item['current_date'],n)
        result = {
            'win_rate_by_coin': win_rate,
            'wins': wins,
            'total_positions': total_positions,
            'avg_growth_rate': avg_growth_rate}
        win_rate_data.append(result)
        dataframes.append(strategy_dataframe)
    return win_rate_data, dataframes
def merge_data_by_strategy_model(strategies_data, strategy_idx, model_idx):
    terms_str = ', '.join(["'" + str(term).lower() + "'" for term in unique_coins.tolist()])
    engine = create_engine("mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer")
    sql = f'''
    SELECT time, prices, slug, symbol
    FROM market_hourly_data_final
    WHERE LOWER(slug) IN ({terms_str}) OR LOWER(symbol) IN ({terms_str})
    '''
    market_data = pd.read_sql_query(sql, engine)
    market_data['slug'] = market_data['slug'].str.lower()
    market_data['symbol'] = market_data['symbol'].str.lower()
    strategy_model_data = strategies_data[strategy_idx][model_idx]
    merged_data_slug = pd.merge(
        market_data, 
        strategy_model_data, 
        left_on=['time', 'slug'], 
        right_on=['current_date', 'coins'], 
        how='left'
    )
    
    # 删除第一次已经匹配的数据
    market_data_unmatched = market_data.loc[~market_data['time'].isin(merged_data_slug['time'][~pd.isna(merged_data_slug['current_date'])])]
    
    # 第二次合并
    merged_data_symbol = pd.merge(
        market_data_unmatched, 
        strategy_model_data, 
        left_on=['time', 'symbol'], 
        right_on=['current_date', 'coins'], 
        how='left'
    )
    final_merged_data = pd.concat([merged_data_slug, merged_data_symbol])
    return final_merged_data.drop_duplicates(subset= ['time','prices_x', 'slug','symbol','coins'])
def merge_data_by_predictions_records(market_data,predictions_records, model_idx):
    prediction_record = predictions_records[model_idx]
    merged_data_slug = pd.merge(
        market_data, 
        prediction_record, 
        left_on=['time', 'slug','prices'], 
        right_on=['current_date', 'coins', 'prices'], 
        how='left'
    )
    
    # 删除第一次已经匹配的数据
    market_data_unmatched = market_data.loc[~market_data['time'].isin(merged_data_slug['time'][~pd.isna(merged_data_slug['current_date'])])]
    
    # 第二次合并
    merged_data_symbol = pd.merge(
        market_data_unmatched, 
        prediction_record, 
        left_on=['time', 'symbol','prices'], 
        right_on=['current_date', 'coins','prices'], 
        how='left'
    )
    final_merged_data = pd.concat([merged_data_slug, merged_data_symbol])
    final_merged_data = final_merged_data.drop_duplicates(subset= ['time','prices', 'slug','symbol','coins'])
    return final_merged_data
def get_win_rates_and_cases(results):
    win_rate_dataset = []
    strategy_dataset = []
    for n in [1,2,3,5,10]:
        win_rate_data,strategy_data = get_win_rate(results, n)
        win_rate_dataset.append(win_rate_data)
        strategy_dataset.append(strategy_data)
    return win_rate_dataset, strategy_dataset
#%%
def plot_model_performance(data,files):
    metrics_to_plot_1 = ["Accuracy", "ROC AUC", "Precision", "Recall"]
    for metric in metrics_to_plot_1:
        values = [r[metric] for r in data]
        mean_value = np.mean(values)
        # 为每个指标创建一个新的图
        plt.figure(figsize=(16, 8), dpi = 400)
        plt.plot(values, label=metric, marker='o')
        plt.axhline(y=mean_value, color='orange', linestyle='-', label=f"Mean {metric} = {mean_value:.2f}")
        plt.xticks(range(0,len(files), 5),[str(extract_number(files[i])) for i in range(0, len(files), 5)], rotation=45) 
        plt.legend()
        plt.title(f"{metric} Over Different Datasets")
        plt.xlabel("Time Window")
        plt.ylabel(metric)
        plt.show()   
def plot_win_rate(data,files):
    metrics_to_plot_2 = ["win_rate_by_coin"]
    for win_rate_data in data:
        for metric in metrics_to_plot_2:
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 16), dpi=400, gridspec_kw={'height_ratios': [3, 3]})
            fig.tight_layout(pad=3.0)  # Add some space between the plots
    
            # 绘制win_rate_by_coin on ax1
            values_win_rate = [r["win_rate_by_coin"] for r in win_rate_data]
            ax1.plot(values_win_rate, label="win_rate_by_coin", marker='o', color='blue')
            ax1.set_ylabel('Win Rate', color='blue')
            ax1.tick_params('y', colors='blue')
            ax1.set_title("Win Rate Over Different Datasets")
            ax1.set_xlabel("Time Window")
            
            avg_win_rate = sum(values_win_rate) / len(values_win_rate)
            ax1.axhline(avg_win_rate, color='orange', linestyle='--', label=f'Avg win_rate: {avg_win_rate:.2f}')
    
            ax2 = ax1.twinx()
            values_wins = [r["wins"] for r in win_rate_data]
            values_total_positions = [r["total_positions"] for r in win_rate_data]
            ax2.plot(values_wins, label="wins", marker='x', linestyle='--', color='green')
            ax2.plot(values_total_positions, label="total_positions", marker='.', linestyle='-.', color='red')
            ax2.set_ylabel('Count', color='black')
            
            ax1.set_xticks(range(0, len(files), 5))
            ax1.set_xticklabels([str(extract_number(files[i])) for i in range(0, len(files), 5)], rotation=45)
    
            # Subplot for avg_growth_rate on ax3
            avg_growth_rates = [r["avg_growth_rate"] for r in win_rate_data]  # Assuming you have this data in your win_rate_data
            ax3.plot(avg_growth_rates, label="median_growth_rate", marker='s', color='purple')
            ax3.set_ylabel('Median Growth Rate', color='purple')
            ax3.set_xticks(range(0, len(files), 5))
            ax3.set_xticklabels([str(extract_number(files[i])) for i in range(0, len(files), 5)], rotation=45)
            ax3.set_xlabel("Time Window")
    
            # Legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)
            
            plt.show()

#%%
# sort files
def sort_files(filepath, features_type):
    folder = os.listdir(filepath+features_type)
    folder = sorted(folder,key = extract_number)
    files = []
    for file in folder:
        file = file.replace('.pickle', '')
        files.append(file)
    return files
#%%
def plot_buy_and_sell(merged_df, selected_coins=None, n=1):
    if selected_coins is None:
        selected_coins = merged_df['coins'].dropna().unique().tolist()

    valid_combinations = []

    for coin in selected_coins:
        coin_data = merged_df[(merged_df['slug'] == coin) | (merged_df['symbol'] == coin)]

        # Group by slug and symbol
        for (slug, symbol), group in coin_data.groupby(['slug', 'symbol']):
            # Check if there's buy data for this slug and symbol combination
            initial_buy_data = group.dropna(subset=['current_date', 'prices'])
            buy_data = initial_buy_data[initial_buy_data['double_prices'].isin(group['prices'])]
            if len(buy_data) != 0:
                valid_combinations.append((slug, symbol))

    n_coins = len(valid_combinations)
    if n_coins == 0:
        print("No valid data to plot!")
        return

    coins_per_figure = math.ceil(n_coins / n)
    
    for i in range(0, n_coins, coins_per_figure):
        current_combinations = valid_combinations[i:i + coins_per_figure]
        fig, axes = plt.subplots(nrows=len(current_combinations), figsize=(20, 5*len(current_combinations)), dpi=200)

        if len(current_combinations) == 1:
            axes = [axes]

        for ax, (slug, symbol) in zip(axes, current_combinations):
            coin_data = merged_df[(merged_df['slug'] == slug) & (merged_df['symbol'] == symbol)]
            buy_data = coin_data.dropna(subset=['current_date', 'prices'])

            sns.lineplot(x='time', y='prices', data=coin_data, ax=ax, label='Price Trend')
            sns.scatterplot(x='current_date', y='prices', data=buy_data[buy_data['status'] == 'Success'], color='green', s=200, label='Buy', ax=ax)
            sns.scatterplot(x='current_date', y='prices', data=buy_data[buy_data['status'] == 'Fail'], color='black', s=200, label='Failed Buy', ax=ax)
            double_data = coin_data.dropna(subset=['double_dates', 'double_prices'])
            if len(double_data) != 0:
                sns.scatterplot(x='double_dates', y='double_prices', data=double_data[double_data['status'] == 'Success'], color='red', s=200, label='Double Point', ax=ax)
            texts = []  # 保存所有的文本标签
            growth_rate_data = buy_data[buy_data['status'] == 'Success']
            for _, row in growth_rate_data.iterrows():
                t = ax.text(row['current_date'], row['prices'], f"Growth: {row['growth_rates']*100:.2f}%", verticalalignment='bottom')
                texts.append(t)
            if len(texts) >= 15:
                texts = texts[:15]
            # 调整文本标签以减少重叠
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='orange'))
            ax.set_title(f"Buy and Double for {slug} ({symbol})")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()

        plt.tight_layout()
        plt.show()

 #%%
def get_market_data(unique_coins):
    terms_str = ', '.join(["'" + str(term).lower() + "'" for term in unique_coins.tolist()])
    engine = create_engine("mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer")
    sql = f'''
    SELECT time, prices, slug, symbol
    FROM market_hourly_data_final
    WHERE LOWER(slug) IN ({terms_str}) OR LOWER(symbol) IN ({terms_str})
    '''
    market_data = pd.read_sql_query(sql, engine)
    market_data['slug'] = market_data['slug'].str.lower()
    market_data['symbol'] = market_data['symbol'].str.lower()
    return market_data
#%%
filepath = r"./dataset/en/"
features_type = "multi/"
threshold = "0.1"
testing_pct = 0.5

results, unique_coins= get_lightgbm_results(filepath, features_type, testing_pct, threshold)
win_rate_dataset, _ = get_win_rates_and_cases(results)

files = sort_files(filepath, features_type)
plot_model_performance(results,files)
plot_win_rate(win_rate_dataset, files)

predictions_records = get_model_predictions_records(results)
#cases_data = merge_data_by_strategy_model(strategy_dataset, strategy_idx, model_idx, unique_coins)
#%%
market_data = get_market_data(unique_coins)
strategy_idx = 0 # 0 - 4 表示策略1，2，3，5，10 （第n次预测翻倍建仓）
model = [0, 20, 30, 50, 110] # 0 - 110 表示时间窗口5-60天，间隔半天
for model_idx in model:
    cases_data = merge_data_by_predictions_records(market_data,predictions_records, model_idx)
    plot_buy_and_sell(cases_data, n = 3)
