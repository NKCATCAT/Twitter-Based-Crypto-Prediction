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

## DualGCNbert
### Active_Learning_Pytorch.py
#### 模型参数
##### Overall Parameters

| Parameter                  | Default Value                                         | Type    | Description                          |
|----------------------------|-------------------------------------------------------|---------|--------------------------------------|
| `--iteration`              | 0                                                     | int     | Num of iteration 主动学习轮次         |
| `--model_name`             | dualgcnbert                                           | str     | model class 选用模型                  |
| `--dataset`                | twitter                                               | str     | dataset_files                        |
| `--optimizer`              | adam                                                  | str     | optimizers                           |
| `--initializer`            | xavier_uniform_                                       | str     | ...                                  |
| `--num_epoch`              | 5                                                     | int     | ...                                  |
| `--batch_size`             | 24                                                    | int     | ...                                  |
| `--log_step`               | 20                                                    | int     | ...                                  |
| `--polarities_dim`         | 3                                                     | int     | 情感极性种类                          |
| `--device`                 | cuda                                                  | str     | cpu or cuda                          |
| `--seed`                   | 1000                                                  | int     | ...                                  |
| `--weight_decay`           | 0                                                     | int     | weight_decay if applied              |
| `--loss_type`              | doubleloss                                            | str     | doubleloss, orthogonal, differentiated|
| `--alpha`                  | 0.5                                                   | float   | ...                                  |
| `--beta`                   | 0.9                                                   | float   | ...                                  |


##### GCN Configuration

| Parameter                  | Default Value                                         | Type    | Description                          |
|----------------------------|-------------------------------------------------------|---------|--------------------------------------|
| `--num_layers`             | 2                                                     | int     | Num of GCN layers.                   |
| `--gcn_dropout`            | 0.1                                                   | float   | GCN layer dropout rate.              |
| `--learning_rate`          | 0.001                                                 | float   | GCN lr                               |
| `--attention_heads`        | 1                                                     | int     | multi-attention heads for SEMGCN     |
| `--direct`                 | False                                                 |         | directed graph or undirected graph   |
| `--hidden_dim`             | 768                                                   | int     | GCN dim 为hidden_dim的一半            |
| `--parseadj`               | True                                                  |         | dependency probability               |


##### BERT Configuration

| Parameter                  | Default Value                                         | Type    | Description                          |
|----------------------------|-------------------------------------------------------|---------|--------------------------------------|
| `--pretrained_bert_name`   | bert-base-uncased                                     | str     |                                      |
| `--bert_dim`               | 768                                                   | int     |                                      |
| `--bert_dropout`           | 0.3                                                   | float   | BERT dropout rate.                   |
| `--adam_epsilon`           | 1e-6                                                  | float   | Epsilon for Adam optimizer.          |
| `--diff_lr`                | False                                                 |         | if different lr is used              |
| `--bert_lr`                | 2e-5                                                  | float   |                                      |
| `--max_length`             | 150                                                   | int     |max sequence length                   |

##### Azure Labeling

| Parameter                  | Default Value                                         | Type    | Description                          |
|----------------------------|-------------------------------------------------------|---------|--------------------------------------|
| `--connection_string`      | <connection_string_value>                             | str     |                                      |
| `--ws_subscrip_id`         | <subscription_id_value>                               | str     |                                      |
| `--container_name`         | <container_name_value>                                | str     |                                      |
| `--resource_group`         | <resource_group_value>                                | str     |                                      |
| `--ws_name`                | <workspace_value>                                     | str     |                                      |
