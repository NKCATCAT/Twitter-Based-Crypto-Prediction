"""
Created on Thu Sep  7 11:13:26 2023

@author: 86189
"""
from datetime import datetime
import numpy as np
import pandas as pd
decay_factor = 0.00005
date_string = '2023-08-16 00:00:00'
def temperature(data):
    date_max = datetime.strptime(date_string,"%Y-%m-%d %H:%M:%S")

    data['hours_from_inspect'] = (date_max - data['date']).dt.total_seconds() / 3600
    data['time_decay'] = np.exp(-decay_factor * data['hours_from_inspect'])

    # 定义bins
    label_likes = np.linspace(0, 1, 20).tolist()
    label_retweets = np.linspace(0, 1, 3).tolist()
    label_replies = np.linspace(0, 1, 3).tolist()
    label_views = np.linspace(0, 1, 5000).tolist()
    
    data['likes_ranking'] = pd.cut(data['likes'], bins = list(range(21)) + [np.inf], labels = label_likes + [1], ordered= False,include_lowest=True).astype(float)
    data['replies_ranking'] = pd.cut(data['replies'], bins = list(range(4)) + [np.inf], labels = label_replies + [1], ordered= False,include_lowest=True).astype(float)
    data['retweets_ranking'] = pd.cut(data['retweets'], bins = list(range(4)) + [np.inf], labels = label_retweets + [1], ordered= False,include_lowest=True).astype(float)
    data['views_ranking'] = pd.cut(data['views'], bins = list(range(5001)) + [np.inf], labels = label_views + [1], ordered= False,include_lowest=True).astype(float)
    data['temperature'] = (data['likes_ranking'] + data['replies_ranking'] + 
                                 data['retweets_ranking'] + data['views_ranking']) * data['time_decay']
    data = data.drop(columns = ['likes_ranking', 'replies_ranking', 'retweets_ranking', 'views_ranking','time_decay','hours_from_inspect'])
    return data