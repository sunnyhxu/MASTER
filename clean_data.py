import pandas as pd
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import pickle 
import argparse
from tqdm import tqdm
import os 
import psutil
path = '../Downloads/total_data.csv'
attributes = ['fcf_be', 'div12m_me', 'me_company','debt_at','beta_21d','qmj_prof', 'qmj_growth',
               'qmj_safety','ret','id','date','ret_6_1','ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_12_1', 'ret_12_7',
               'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na', 'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 'seas_11_15na', 'seas_16_20an', 'seas_16_20na', 'at_gr1', 'sale_gr1', 'capx_gr1', 'inv_gr1', 'debt_gr3', 'sale_gr3', 'capx_gr3', 'inv_gr1a', 'lti_gr1a', 'sti_gr1a', 'coa_gr1a', 'col_gr1a', 'cowc_gr1a', 'ncoa_gr1a', 'ncol_gr1a', 'nncoa_gr1a',
               'ebit_bev', 'netis_at',  'sale_emp_gr1', 'emp_gr1', 'ni_inc8q', 'noa_gr1a', 'ppeinv_gr1a', 'lnoa_gr1a', 'capx_gr2', 'at_be',
               'debt_me', 'ival_me', 'bev_mev', 'ebitda_mev', 'aliq_mat', 'eq_dur', 'beta_60m', 'resff3_12_1', 'resff3_6_1', 'ivol_capm_21d', 'iskew_capm_21d', 'coskew_21d', 'beta_dimson_21d', 'ivol_ff3_21d', 'iskew_ff3_21d', 'ivol_hxz4_21d', 'iskew_hxz4_21d',
                'fcf_mev', 'debt_mev', 'pstk_mev', 'debtlt_mev', 'debtst_mev', 'dltnetis_mev', 'dstnetis_mev', 'dbnetis_mev', 'netis_mev', 'fincf_mev', 'ivol_capm_60m', 'beta_252d', 'rvol_252d', 'rvolhl_21d','ret_exc_lead1m']
short_att= ['fcf_be', 'div12m_me','beta_21d','qmj_prof', 'qmj_growth','seas_1_1an', 'seas_2_5an', 'seas_2_5na', 'seas_6_10an',
               'qmj_safety','ret','id','date','ret_6_1','ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_9_1','ret','ret_exc_lead1m']

def process_data(file_path:str, list_attr:list,train_perc:float, thresh=10, time_periods=4):
    num_stocks = 0
    train = []
    valid = []
    test = []
    a = 1
    for total_stock in pd.read_csv(file_path, chunksize=400000, memory_map=True):
        print(f'starting section a={a}')
        # restricting data to common stock and primary exchanges 
        total_stock = total_stock.loc[(total_stock['common']==1) & total_stock['exch_main']==1]

        shortended_df = total_stock[list_attr]
        # dropping any nan values to make sure each observation has all necessary fields of data
        shortended_df = shortended_df.fillna(0)

        groups = shortended_df.groupby(['id'])
        unique_ids = shortended_df['id'].unique().tolist()

        cutoff = []
        # starting the stock aggregation loop to aggregate data by stocks 
        for i in tqdm(range(len(unique_ids))):
            num = unique_ids[i]
            count = groups.get_group(num)['ret'].count()
            
            if count[0] >= thresh:
                cutoff.append(num)
        print(f'stock cutoffs completed and selected {len(cutoff)} stocks ')

        
        print("Entering stock slicing section")
        for i in tqdm(range(len(cutoff))):
          
            num = cutoff[i]
            # Curr_df is a (time period, feature length) matrix where time period is >= threshold 
            curr_df = groups.get_group(num)
            returns  = curr_df['ret_exc_lead1m'].to_numpy()
            num_stocks+=1
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
        print(f'{200000*a} entries complete , num stocks{num_stocks}')
        a+=1
       
    
    return train, valid, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', required=True)
    arg = vars(parser.parse_args())
    file_name = arg['file_name']

    # calling clean function 
    train_data, valid_data, test_data = process_data(file_name, short_att, 0.8)
    with open('data/largepickle_train2', 'wb') as file:
        pickle.dump(train_data,file )
        file.close()
    with open('data/largepickle_validate2', 'wb') as file:
        pickle.dump(valid_data, file); 
        file.close()
    with open('data/largepickle_test2', 'wb') as file:
        pickle.dump(test_data, file)
        file.close()
        
    
if __name__ == "__main__":
    main()



