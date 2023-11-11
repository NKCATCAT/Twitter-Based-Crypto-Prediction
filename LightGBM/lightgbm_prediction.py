#%%
import lightgbm as lgb
import pandas as pd
import pickle
from sqlalchemy import create_engine
#%%
model_path = "./models/model_876.txt"
classifier_path = "./dataset/en/multi/10%/multi_36.5days_10%.pickle"
def predict_lightgbm_inputs(model_path, classifier_path):
    with open(classifier_path,'rb') as f:
        classifier_inputs = pickle.load(f)       
    classifier_inputs = classifier_inputs[classifier_inputs['T_876h'] > 0]                          
    all_columns = set(classifier_inputs.columns)
    exclude_columns = set(['date', 'name', 'symbol', 'pool_id', 'price'])
    features = list(all_columns - exclude_columns)
    
    X_pre = classifier_inputs[features]
    bst = lgb.Booster(model_file=model_path)
    y_pred_prob = bst.predict(X_pre)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]
    
    result_df = classifier_inputs[['date', 'name', 'symbol', 'pool_id', 'price']].copy()
    result_df = result_df.rename(columns = {"date":"signal_date"})
    result_df['is_double'] = y_pred
    return result_df
def saveto_database(dataframe, database_tosave, sql_connector):
    engine = create_engine(sql_connector)
    sql_cmd = f'''SELECT * FROM {database_tosave}'''  
    try:
        existing_data = pd.read_sql(sql_cmd, engine)
    except:
        existing_data = pd.DataFrame(columns=['signal_date', 'pool_id', 'is_double'])
    merged_data = existing_data.merge(dataframe, on=['signal_date', 'pool_id'], how='outer', suffixes=('', '_new'))
    rows_to_update = merged_data.loc[
    (~merged_data['is_double'].isnull()) & 
    (~merged_data['is_double_new'].isnull()) & 
    (merged_data['is_double'] != merged_data['is_double_new'])]
    for index, row in rows_to_update.iterrows():
        update_cmd = f"""UPDATE {database_tosave} SET is_double = {row['is_double_new']} WHERE signal_date = '{row['signal_date']}' AND pool_id = '{row['pool_id']}'"""
        engine.execute(update_cmd)
    rows_to_insert = merged_data[merged_data['is_double'].isnull()]
    if not rows_to_insert.empty:
        rows_to_insert = rows_to_insert.drop(columns=['is_double'])
        rows_to_insert = rows_to_insert.rename(columns={'is_double_new': 'is_double'})
        rows_to_insert.to_sql(database_tosave, engine, index=False, if_exists='append')
#%%
database_tosave = "signal_tokens"
sql_connector = "mysql+mysqlconnector://tangshuo:tangshuo@121.36.100.76:13310/ai_summer"
dataframe = predict_lightgbm_inputs(model_path, classifier_path)
saveto_database(dataframe, database_tosave, sql_connector)