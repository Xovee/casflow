import copy
import math

import numpy as np
import tensorflow as tf
from tensorflow.python import keras


EPSILON = 1e-7


class Generator(keras.utils.Sequence):

    def __init__(self, input_vae, input_global, label, b_size, max_length):
        self.vae, self.input_global, self.y = input_vae, input_global, label
        self.batch_size = b_size
        self.max_cascade_length = max_length

    def __len__(self):
        return math.ceil(len(self.vae)/self.batch_size)   # ceil or floor

    def __getitem__(self, idx):
        b_vae = copy.deepcopy(
            self.vae[idx*self.batch_size:(idx+1)*self.batch_size])
        b_global = copy.deepcopy(
            self.input_global[idx * self.batch_size:(idx + 1) * self.batch_size])
        b_y = copy.deepcopy(self.y[idx*self.batch_size:(idx+1)*self.batch_size])
        for vae in b_vae:
            while len(vae) < self.max_cascade_length:
                vae.append(np.zeros(shape=len(vae[0])))
        for glo in b_global:
            while len(glo) < self.max_cascade_length:
                glo.append(np.zeros(shape=len(glo[0])))

        b_x = np.concatenate([np.array(b_vae), np.array(b_global)], axis=2)

        return b_x, np.array(b_y)


class Sampling(keras.layers.Layer):

    def call(self, mean_log_var, **kwargs):
        mean, log_var = mean_log_var
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        EPSILON = keras.backend.random_normal(shape=(batch, dim))

        return mean + tf.exp(.5 * log_var) * EPSILON


class VAE(keras.layers.Layer):
    def __init__(self,
                 emb_dim,
                 h_dim,
                 seq_length,
                 gru_units):
        super(VAE, self).__init__()
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.seq_length = seq_length
        self.gru_units = gru_units

        self.inference_net = tf.keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.seq_length, self.h_dim)),
            keras.layers.Bidirectional(
                keras.layers.CuDNNGRU(gru_units*2, return_sequences=True),
            ),
            keras.layers.Bidirectional(
                keras.layers.CuDNNGRU(gru_units),
            ),
            keras.layers.Dense(self.h_dim*2),
        ])

        self.generative_net = tf.keras.Sequential([
            keras.layers.RepeatVector(self.seq_length),
            keras.layers.CuDNNGRU(self.gru_units, return_sequences=True),
            keras.layers.TimeDistributed(
                keras.layers.Dense(self.emb_dim)
            )
        ])

    def node_encoder(self, x):
        x = tf.reshape(x, shape=(-1, self.emb_dim))
        mean = keras.layers.Dense(self.h_dim)(x)
        log_var = keras.layers.Dense(self.h_dim)(x)
        return mean, log_var

    def node_decode(self, z):
        z = keras.layers.Dense(self.emb_dim)(z)
        reconstruct_x = tf.reshape(z, shape=(-1, self.seq_length, self.emb_dim))
        return reconstruct_x

    def encoder(self, x):
        mean, log_var = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    def decode(self, z):
        reconstruct_x = self.generative_net(z)
        return reconstruct_x


def nf_transformations(z, dim, k):
    z0 = z
    logD_loss = 0

    zk, logD = PlanarFlowLayer1(dim)(z0)

    for i in range(k):
        zk, logD = PlanarFlowLayer(dim)((zk, logD))
        logD_loss += logD

    return zk, logD_loss


class PlanarFlowLayer1(keras.layers.Layer):
    def __init__(self,
                 z_dim):
        super(PlanarFlowLayer1, self).__init__()
        self.z_dim = z_dim

        self.w = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.u = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        z_prev = inputs
        m = lambda x: -1 + tf.math.log(1 + tf.exp(x))
        h = lambda x: tf.tanh(x)
        h_prime = lambda x: 1 - h(x) ** 2
        u_hat = (m(tf.tensordot(self.w, self.u, 2)) - tf.tensordot(self.w, self.u, 2)) \
                     * (self.w / tf.norm(self.w)) + self.u
        z_prev = z_prev + u_hat * h(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b)
        affine = h_prime(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b) *self.w
        sum_log_det_jacob = tf.math.log(EPSILON + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))

        return z_prev, sum_log_det_jacob


class PlanarFlowLayer(keras.layers.Layer):
    def __init__(self,
                 z_dim):
        super(PlanarFlowLayer, self).__init__()
        self.z_dim = z_dim

        self.w = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.u = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        z_prev, sum_log_det_jacob = inputs
        m = lambda x: -1 + tf.math.log(1 + tf.exp(x))
        h = lambda x: tf.tanh(x)
        h_prime = lambda x: 1 - h(x) ** 2
        u_hat = (m(tf.tensordot(self.w, self.u, 2)) - tf.tensordot(self.w, self.u, 2)) \
                * (self.w / tf.norm(self.w)) + self.u
        z_prev = z_prev + u_hat * h(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b)
        affine = h_prime(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b) * self.w
        sum_log_det_jacob += tf.math.log(EPSILON + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))

        return z_prev, sum_log_det_jacob

