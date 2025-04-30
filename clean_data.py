import tensorflow as tf 
import numpy as np 
import pandas as pd 
import pickle 
import argparse
from tqdm import tqdm


attributes = ['fcf_be', 'div12m_me', 'me_company','debt_at','beta_21d','qmj_prof', 'qmj_growth',
               'qmj_safety','ret','id','date','ret_6_1','ret_exc_lead1m']

def process_data(file_path:str, list_attr:list,train_perc:float, thresh=7, time_periods=3):
    total_stock = pd.read_csv(file_path, memory_map=True, low_memory=True, skiprows=[i for i in range(250000,685000)])
    # restricting data to common stock and primary exchanges 
    total_stock = total_stock.loc[(total_stock['common']==1) & total_stock['exch_main']==1]

    shortended_df = total_stock[list_attr]
    # dropping any nan values to make sure each observation has all necessary fields of data
    shortended_df = shortended_df.dropna(axis=0, how='any')
    print(shortended_df.columns)

    groups = shortended_df.groupby(['id'])
    unique_ids = shortended_df['id'].unique().tolist()

    cutoff = []
    # starting the stock aggregation loop to aggregate data by stocks 
    for i in tqdm(range(len(unique_ids))):
        num = unique_ids[i]
        count = groups.get_group(num)['ret'].count()
        
        if count >= thresh:
            cutoff.append(num)
    print(f'stock cutoffs completed and selected {len(cutoff)} stocks ')

    train = []
    valid = []
    test = []
    print(len(cutoff))
    print("Entering stock slicing section")
    for i in tqdm(range(len(cutoff))):
        num = cutoff[i]
        # Curr_df is a (time period, feature length) matrix where time period is >= threshold 
        curr_df = groups.get_group((num,))
        returns  = curr_df['ret_exc_lead1m'].to_numpy()

        # Trimming curr_df so all stocks have the same lookback period 
        curr_df = curr_df.iloc[0:thresh, :]
        obs = curr_df.drop(columns=['ret_exc_lead1m','date']).to_numpy()

        # standardizing by the mean of the features and the returns 
        feature_max = tf.math.reduce_max(obs, axis=0)
        return_max  = tf.math.reduce_max(returns)
        returns:tf.Tensor = tf.math.divide(returns, 1)
        feature:tf.Tensor= tf.math.divide(obs, 1)

        # breaking it down into train test and validation sets
        index = int(np.floor(feature.shape[0]*train_perc))
        index = index - (index % time_periods)
        iters = int(index / time_periods)
        train_feature = feature[:index, :]
        train_returns = returns[:index]
        
        # adding the training data 
        for i in range(0, iters):
            curr_feat = feature[i*time_periods:time_periods + i*time_periods,:]
            curr_ret = returns[i*time_periods:time_periods + i*time_periods]
            train.append((curr_feat,curr_ret))
        
        # validation can be a small set
        validate_feature = feature[:time_periods, :]
        validate_returns = returns[:time_periods]
        valid.append((validate_feature, validate_returns))

        # Now assembling the test test 
        test_feature  =feature[index:,:]
        test_rets = returns[index:]
        test_num = test_rets.shape[0]
        test_iters = int(np.floor(test_num/time_periods))
         
        
        # Establishing the test set - note that the number of periods in the test set should 
        # equal the number of periods in the training set as we want our model to learn 
        # to predict given a set of number of months 
        for j in range(0, test_iters):
            feat_test = test_feature[i*time_periods:time_periods+i*time_periods,:]
            ret_test = test_rets[i*time_periods:time_periods+i*time_periods]
            test.append((feat_test, ret_test))


       
    
    return train, valid, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', required=True)
    arg = vars(parser.parse_args())
    file_name = arg['file_name']

    # calling clean function 
    train_data, valid_data, test_data = process_data(file_name, attributes, 0.8, )
    with open('data/pickle_train', 'wb') as file:
        pickle.dump(train_data,file )
        file.close()
    with open('data/pickle_validate', 'wb') as file:
        pickle.dump(valid_data, file); 
        file.close()
    with open('data/pickle_test', 'wb') as file:
        pickle.dump(test_data, file)
        file.close()
        
    
if __name__ == "__main__":
    main()





