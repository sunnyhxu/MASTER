import numpy as np
import tensorflow as tf
import keras
import math

from base_model import SequenceModel

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, d_model, max_len=100, **kwargs):
        super().__init__(**kwargs)
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:tf.shape(inputs)[1], :]
    

# Inter-Stock Aggregation
class SAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dim = d_model // num_heads
        # to deal with the case when d_model is not divisible by num_heads
        self.remainder = d_model % num_heads
        self.dropout_rate = dropout_rate

        self.q_dense = keras.layers.Dense(d_model, use_bias=False)
        self.k_dense = keras.layers.Dense(d_model, use_bias=False)
        self.v_dense = keras.layers.Dense(d_model, use_bias=False)

        self.attn_dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(num_heads)]
        
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        
        self.ffn = keras.Sequential([
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=None):
        # inputs shape: (N, T, D)

        normed_inputs = self.norm1(inputs)

        q = tf.transpose(self.q_dense(normed_inputs), perm=[1, 0, 2])
        k = tf.transpose(self.k_dense(normed_inputs), perm=[1, 0, 2])
        v = tf.transpose(self.v_dense(normed_inputs), perm=[1, 0, 2])
        # q, k, v shape: (T, N, D)

        attention_output = []
        cur_idx = 0
        for i in range(self.num_heads):
            # different implementation of head dimension than the original paper
            if i < self.remainder:
                cur_dim = self.dim + 1
            else:
                cur_dim = self.dim

            temperature = math.sqrt(cur_dim)
            end_idx = cur_idx + cur_dim
            q_i = q[:, :, cur_idx:end_idx]
            k_i = k[:, :, cur_idx:end_idx]
            v_i = v[:, :, cur_idx:end_idx]
            # shape: (T, N, D/N) or (T, N, D/N + 1)

            attention_weights = tf.matmul(q_i, k_i, transpose_b=True) / temperature  # (T, N, N)
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)

            if training:
                attention_weights = self.attn_dropout_layers[i](attention_weights, training=training)

            attention_output.append(tf.transpose(tf.matmul(attention_weights, v_i), perm=[1, 0, 2])) 

            cur_idx = end_idx
        
        concat_output = tf.concat(attention_output, axis=-1)  # (N, T, D)
        attention_output = self.norm2(concat_output + inputs)
        ffn_output = self.ffn(attention_output, training=training)

        return attention_output + ffn_output


# Intra-Stock Aggregation
class TAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dim = d_model // num_heads
        # to deal with the case when d_model is not divisible by num_heads
        self.remainder = d_model % num_heads
        self.dropout_rate = dropout_rate

        self.q_dense = keras.layers.Dense(d_model, use_bias=False)
        self.k_dense = keras.layers.Dense(d_model, use_bias=False)
        self.v_dense = keras.layers.Dense(d_model, use_bias=False)

        self.attn_dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(num_heads)]

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.ffn = keras.Sequential([
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=None):
        # inputs shape: (N, T, D)

        normed_inputs = self.norm1(inputs)

        q = self.q_dense(normed_inputs)
        k = self.k_dense(normed_inputs)
        v = self.v_dense(normed_inputs)
        # q, k, v shape: (N, T, D)

        attention_output = []
        cur_idx = 0
        for i in range(self.num_heads):
            # different implementation of head dimension than the original paper
            if i < self.remainder:
                cur_dim = self.dim + 1
            else:
                cur_dim = self.dim

            end_idx = cur_idx + cur_dim
            q_i = q[:, :, cur_idx:end_idx]
            k_i = k[:, :, cur_idx:end_idx]
            v_i = v[:, :, cur_idx:end_idx]
            # shape: (N, T, D/N) or (N, T, D/N + 1)

            attention_weights = tf.matmul(q_i, k_i, transpose_b=True) # No temperature scaling
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)

            if training:
                attention_weights = self.attn_dropout_layers[i](attention_weights, training=training)

            attention_output.append(tf.matmul(attention_weights, v_i))

            cur_idx = end_idx
        
        concat_output = tf.concat(attention_output, axis=-1)  # (N, T, D)
        attention_output = self.norm2(concat_output + inputs)
        ffn_output = self.ffn(attention_output, training=training)

        return attention_output + ffn_output


# Market-Guided Gating
class Gate(keras.layers.Layer):
    def __init__(self, d_input, d_output, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d_input = d_input
        self.d_output = d_output
        self.beta = beta
        self.dense = keras.layers.Dense(d_output)

    def call(self, inputs):
        output = self.dense(inputs)
        output = tf.nn.softmax(output / self.beta, axis=-1)
        return self.d_output * output
    

# Temporal Aggregation
class TemporalAttention(keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dense = keras.layers.Dense(d_model, use_bias=False)
    
    def call(self, inputs):
        h = self.dense(inputs)  # (N, T, D)
        query = h[:, -1:, :]  # (N, 1, D)
        lam = tf.matmul(h, query, transpose_b=True)  # (N, T, 1)
        lam = tf.nn.softmax(lam, axis=-1)

        output = tf.matmul(tf.transpose(lam, perm=[0, 2, 1]), inputs)  # (N, 1, T) x (N, T, D) = (N, 1, D)
        return tf.squeeze(output, axis=1)  # (N, D)


# MASTER
class MASTER(keras.Model):
    def __init__(self, d_feat, d_model, t_num_heads, s_num_heads, t_dropout_rate, s_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta, **kwargs):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.t_num_heads = t_num_heads
        self.s_num_heads = s_num_heads
        self.t_dropout_rate = t_dropout_rate
        self.s_dropout_rate = s_dropout_rate
        self.feature_gate = Gate(self.d_gate_input, self.d_model, beta=beta)

        self.master_layers = keras.Sequential([
            # feature layer
            keras.layers.Dense(d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model, t_num_heads, t_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model, s_num_heads, s_dropout_rate),
            # temporal aggregation
            TemporalAttention(d_model),
            # decoder
            keras.layers.Dense(1)
        ])

    def call(self, inputs, training=None):
        # inputs shape: (N, T, F_total) where F_total = features + market_info
        feature_input = inputs[:, :, :self.gate_input_start_index]  # (N, T, D)
        gate_input = inputs[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # (N, D_gate)
        features = feature_input * tf.expand_dims(self.feature_gate(gate_input), axis=1)  # (N, T, D)

        output = self.master_layers(features, training=training)
        return tf.squeeze(output, axis=-1)
    m
class MASTERModel(SequenceModel):
    def __init__(self, d_feat, d_model, t_num_heads, s_num_heads, gate_input_start_index, gate_input_end_index,
                 t_dropout_rate, s_dropout_rate, beta, **kwargs):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.t_dropout_rate = t_dropout_rate
        self.s_dropout_rate = s_dropout_rate
        self.t_num_heads = t_num_heads
        self.s_num_heads = s_num_heads
        self.beta = beta
        
        self.init_model()
    
    def init_model(self):
        self.model = MASTER(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_num_heads=self.t_num_heads,
            s_num_heads=self.s_num_heads,
            t_dropout_rate=self.t_dropout_rate,
            s_dropout_rate=self.s_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta
        )
        
        super(MASTERModel, self).init_model()