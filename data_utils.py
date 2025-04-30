import numpy as np
import pandas as pd
import tensorflow as tf
import random

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

def drop_extreme(x, percent=0.025):
    sorted_indices = tf.argsort(x).numpy()
    n = x.shape[0]
    
    lower_bound = int(n * percent)
    upper_bound = int(n * (1 - percent))
    
    mask = np.zeros(n, dtype=bool)
    mask[sorted_indices[lower_bound:upper_bound]] = True
    
    filtered_x = x[mask]
    return mask, filtered_x

def drop_na(x):
    mask = ~np.isnan(x)
    filtered_x = x[mask]
    return mask, filtered_x


# to be implemented
# note that data is a list of tuples -> [ (features, returns), ...]
def create_tf_dataset(data, shuffle=False, batch_size= 10):

    data_arr = np.zeros(shape=(len(data), data[0][0].shape[0], 12))
   
    final_batches = []
    
    for i in range(0, len(data)):
        nparr = data[i][0].numpy()
        data_arr[i,:, 0:11] = nparr
        data_arr[i,:,11] = data[i][1].numpy()

    if shuffle == True:
        np.random.shuffle(data_arr)
        returns_arr  = np.zeros(shape=(len(data)))
        for i in range(0, len(data)):
            returns_arr[i] = data_arr[i,data[0][0].shape[0]-1,11]
        data_arr = data_arr[:,:,0:11]

        # preparing slices of data 
        stocks_left = data_arr.shape[0] - batch_size
        i =0 
        while (stocks_left >= batch_size):
            curr_feature = data_arr[i*batch_size: batch_size + i*batch_size,:,0:11]
            curr_returns = returns_arr[i*batch_size: batch_size + i*batch_size]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
            i+=1 
            stocks_left = stocks_left - batch_size
        
        if stocks_left < batch_size:
            curr_len = stocks_left
            last_entry =i*batch_size +batch_size
            curr_feature = data_arr[last_entry :last_entry+curr_len,:,0:11]
            curr_returns = returns_arr[last_entry :last_entry+curr_len]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
        return final_batches
    else:
        returns_arr  = np.zeros(shape=(len(data)))
        for i in range(0, len(data)):
            returns_arr[i] = data_arr[i,4,11]
        data_arr = data_arr[:,:,0:11]

        while (stocks_left >= batch_size):
            curr_feature = data_arr[i*batch_size: batch_size + i*batch_size,:,0:11]
            curr_returns = returns_arr[i*batch_size: batch_size + i*batch_size]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
            i+=1 
            stocks_left = stocks_left - batch_size
        
        if stocks_left < batch_size:
            curr_len = stocks_left
            last_entry =i*batch_size +batch_size
            curr_feature = data_arr[last_entry :last_entry+curr_len,:,0:11]
            curr_returns = returns_arr[last_entry :last_entry+curr_len]
            final_batches.append((tf.convert_to_tensor(curr_feature), tf.convert_to_tensor(curr_returns)))
        
        return final_batches


class DailyBatchSamplerRandom:
    def __init__(self):
        pass