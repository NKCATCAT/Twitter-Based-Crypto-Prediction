# Twitter-Based-Crypto-Prediction
## TwitterCrawler
### TextCrawler.py
#### STEP 1: 获取爬推用的账号
文件 all_accounts  
#### STEP 2: 获取KOL的列表
文件 kol_list  
#### STEP 3: 选取tab
Latest 或者 Top  
#### STEP 4：设定爬取的起始时间和结束时间
start_date = datetime(2023, 7, 24) #爬取推文的起始时间  
end_date = datetime.now() #爬取推文的结束时间  
#### STEP 5: 设定一个账户所爬取一个用户的推文时间范围
days_per_call = 100  
#### STEP 6: 设定爬完一个用户所需的账号
例如一个用户爬取6个月的推文，如果days_per_call设定为100则需要两个账号  
第一个账号爬前100天内容，第二个账号爬后100天内容  
那么需要设定i = 2  
#### STEP 7: 存储
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer"  
database_tosave = "tweets_df_final"  

### FamousUserCrawler.py （采用图广度优先搜索方法 BFS）
#### STEP 1: 设定爬取大V起始点
start_point = "CryptoZhaoX"    
#### STEP 2: 获取爬大V用的账号
文件 all_accounts 
#### STEP 3: 存储
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer"  
database_tosave = "popular_accounts_df"
#### STEP 4：爬取深度
默认爬取深度为20
