import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import time

from data_utils import calc_ic, zscore, drop_extreme

class SequenceModel:
    def __init__(self, n_epochs, lr, seed=None, train_stop_loss_thred=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.fitted = -1
        self.model = None
        
        if self.seed is not None:
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)
    
    def init_model(self):
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = keras.losses.MeanSquaredError()
    
    def train_epoch(self, dataset):
        losses = []
        self.model.trainable = True
        
        for data in dataset:
            features, labels = data
            
            mask, filtered_labels = drop_extreme(labels)
            filtered_features = tf.boolean_mask(features, mask)
            normalized_labels = zscore(filtered_labels)
            
            with tf.GradientTape() as tape:
                predictions = self.model(filtered_features, training=True)
                loss = self.loss_fn(normalized_labels, predictions)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            clipped_gradients = [tf.clip_by_value(g, -3.0, 3.0) for g in gradients]
            self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
            losses.append(loss.numpy())
        
        return float(np.mean(losses))
    
    def test_epoch(self, dataset):
        losses = []
        
        self.model.trainable = False
        
        for data in dataset:
            features, labels = data
            normalized_labels = zscore(labels)
            predictions = self.model(features, training=False)
            loss = self.loss_fn(normalized_labels, predictions)
            losses.append(loss.numpy())
        
        return float(np.mean(losses))
    
    def fit(self, train_dataset, valid_dataset):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_dataset)
            self.fitted = epoch

            predictions, metrics = self.predict(valid_dataset)
            end_time = time.time()
            print(f"Epoch {epoch}, train_loss {train_loss:.6f}, valid ic {metrics['IC']:.4f}, "
                  f"icir {metrics['ICIR']:.3f}, rankic {metrics['RIC']:.4f}, rankicir {metrics['RICIR']:.3f}, "
                  f"time: {end_time - start_time:.2f}s")
            
            if train_loss <= self.train_stop_loss_thred:
                print(f"Early stopping at epoch {epoch} with train_loss {train_loss:.6f}")
                break
            
        return self.model
    
    def predict(self, test_dataset):
        print(f'Epoch: {self.fitted}')
        
        self.model.trainable = False
        
        all_preds = []
        all_labels = []
        ic_values = []
        ric_values = []
        
        for data in test_dataset:
            features, labels = data
            
            predictions = self.model(features, training=False).numpy()
            
            all_preds.append(predictions)
            all_labels.append(labels.numpy())
            
            batch_ic, batch_ric = calc_ic(predictions, labels.numpy())
            if not np.isnan(batch_ic):
                ic_values.append(batch_ic)
            if not np.isnan(batch_ric):
                ric_values.append(batch_ric)
        
        predictions = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        
        metrics = {
            'IC': np.mean(ic_values),
            'ICIR': np.mean(ic_values) / (np.std(ic_values) + 1e-8),
            'RIC': np.mean(ric_values),
            'RICIR': np.mean(ric_values) / (np.std(ric_values) + 1e-8)
        }
        
        return predictions, metrics
