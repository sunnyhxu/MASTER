from master import MASTER
import numpy as np
import tensorflow as tf
import os
import keras
import pickle
import time

from master import MASTERModel
from data_utils import create_tf_dataset

def main():
    universe = 'csi300'  # ['csi300', 'csi800', 'us']
    prefix = 'opensource'  # ['opensource', 'globalfactor']
    data_path = f'../data'
    train_pkl_path = os.path.join(data_path, prefix, f'{universe}_dl_train.pkl')
    valid_pkl_path = os.path.join(data_path, prefix, f'{universe}_dl_valid.pkl')
    test_pkl_path = os.path.join(data_path, prefix, f'{universe}_dl_test.pkl')
    print("Loading Data...")
    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_pkl_path, 'rb') as f:
        valid_data = pickle.load(f)
    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)
    
    train_dataset = create_tf_dataset(train_data, shuffle=True)
    valid_dataset = create_tf_dataset(valid_data, shuffle=False)
    test_dataset = create_tf_dataset(test_data, shuffle=False)
    print("Data Loaded.")

    d_feat = 158
    d_model = 256
    t_num_heads = 4
    s_num_heads = 2
    dropout = 0.5
    gate_input_start_index = 158
    gate_input_end_index = 221

    if universe == 'csi300':
        beta = 5
    elif universe == 'csi800':
        beta = 2

    n_epochs = 1
    lr = 1e-5
    train_stop_loss_thred = 0.95

    num_seeds = 1
    seeds = range(num_seeds)
    epoch = 0

    for seed in seeds:
        epoch += 1
        model = MASTERModel(
            d_feat=d_feat,
            d_model=d_model,
            t_num_heads=t_num_heads,
            s_num_heads=s_num_heads,
            t_dropout_rate=dropout,
            s_dropout_rate=dropout,
            beta=beta,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            n_epochs=n_epochs,
            lr=lr,
            seed=seed,
            train_stop_loss_thred=train_stop_loss_thred
        )

        print("Model Created. Start Training...")

        start_time = time.time()
        model.fit(train_dataset, valid_dataset)
        training_time = time.time() - start_time
        print(f"Epoch: {epoch}, Training Time: {training_time:.2f} seconds")

    print("Training completed. Start Testing...")
    print("Testing model...")
    predictions, metrics = model.predict(test_dataset)
    print(f"Test metrics:")
    print(f"IC: {metrics['IC']:.4f}")
    print(f"ICIR: {metrics['ICIR']:.4f}")
    print(f"RIC: {metrics['RIC']:.4f}")
    print(f"RICIR: {metrics['RICIR']:.4f}")
    print(f"Predictions: {predictions[:5]}")

if __name__ == "__main__":
    main()