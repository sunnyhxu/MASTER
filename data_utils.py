import numpy as np
import pandas as pd
import tensorflow as tf

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

def drop_extreme(x, percent=0.025):
    _, sorted_indices = x.sort()
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

def create_tf_dataset(data, shuffle=False):
    pass

class DailyBatchSamplerRandom:
    def __init__(self):
        pass