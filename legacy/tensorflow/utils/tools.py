import copy
import math
import random

import numpy as np
import tensorflow as tf


class Generator(tf.keras.utils.Sequence):

    def __init__(self, input_vae, input_global, label, b_size, max_length, is_train=True):
        self.vae, self.input_global, self.y = input_vae, input_global, label
        self.batch_size = b_size
        self.max_cascade_length = max_length
        self.is_train = is_train

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

        if self.is_train:
            con = list(zip(b_x, np.array(b_y)))
            random.shuffle(con)
            b_x, b_y = zip(*con)

        return np.array(b_x), np.array(b_y)


class Sampling2D(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(.5 * z_log_var) * epsilon


class Sampling3D(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seq = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, seq, dim))

        return z_mean + tf.exp(.5 * z_log_var) * epsilon


def nf_transformations(z, dim, k):
    z0 = z
    logD_loss = 0

    zk, logD = PlanarFlowLayer(dim, True)(z0)

    for i in range(k):
        zk, logD = PlanarFlowLayer(dim, False)((zk, logD))
        logD_loss += logD

    return zk, logD_loss


class PlanarFlowLayer(tf.keras.layers.Layer):
    def __init__(self,
                 z_dim,
                 is_first_layer=True):
        super(PlanarFlowLayer, self).__init__()
        self.z_dim = z_dim
        self.is_first_layer = is_first_layer

        self.w = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.u = self.add_weight(shape=(1, self.z_dim,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        EPSILON = 1e-7

        if self.is_first_layer:
            z_prev = inputs
        else:
            z_prev, sum_log_det_jacob = inputs
        m = lambda x: -1 + tf.math.log(1 + tf.exp(x))
        h = lambda x: tf.tanh(x)
        h_prime = lambda x: 1 - h(x) ** 2
        u_hat = (m(tf.tensordot(self.w, self.u, 2)) - tf.tensordot(self.w, self.u, 2)) \
                * (self.w / tf.norm(self.w)) + self.u
        z_prev = z_prev + u_hat * h(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b)
        affine = h_prime(tf.expand_dims(tf.reduce_sum(z_prev * self.w, -1), -1) + self.b) *self.w
        if self.is_first_layer:
            sum_log_det_jacob = tf.math.log(EPSILON + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))
        else:
            sum_log_det_jacob += tf.math.log(EPSILON + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))

        return z_prev, sum_log_det_jacob