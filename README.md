# Twitter-Based-Crypto-Prediction
## ProjectURL
https://drive.google.com/file/d/1svsLf5vR76WKC0GlBYgnSUDxGBZZEvrR/view?usp=sharing
## DualGCNbert
### Active_Learning_Pytorch.py
#### Model Parameters
##### Overall Parameters

| Parameter                  | Default Value                                         | Type    | Description                          |
|----------------------------|-------------------------------------------------------|---------|--------------------------------------|
| `--iteration`              | 0                                                     | int     | Num of iteration                     |
| `--model_name`             | dualgcnbert                                           | str     | model class                          |
| `--dataset`                | twitter                                               | str     | dataset_files                        |
| `--optimizer`              | adam                                                  | str     | optimizers                           |
| `--initializer`            | xavier_uniform_                                       | str     | ...                                  |
| `--num_epoch`              | 5                                                     | int     | ...                                  |
| `--batch_size`             | 24                                                    | int     | ...                                  |
| `--log_step`               | 20                                                    | int     | ...                                  |
| `--polarities_dim`         | 3                                                     | int     |                                      |
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
| `--hidden_dim`             | 768                                                   | int     | GCN dim 为hidden_dim                 |
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

### DualGCNbert/DualGCNbert/run.sh

