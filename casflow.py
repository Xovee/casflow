from tensorflow.python import keras
import tensorflow as tf
import config
import pickle
import numpy as np
from utils.tools import *
import os


def casflow_loss(y_true, y_pred):
    mse = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    node_ce_loss_x = tf.reduce_mean(
        keras.losses.mean_squared_error(bn_casflow_inputs, node_recon))
    node_kl_loss = - 0.5 * tf.reduce_mean(
        1+node_z_log_var-tf.square(node_z_mean)-tf.exp(node_z_log_var))

    ce_loss_x = tf.reduce_mean(
        keras.losses.mean_squared_error(bn_casflow_inputs, recon_x))
    kl_loss = - 0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return mse + node_ce_loss_x + ce_loss_x + node_kl_loss + kl_loss - logD_loss


with open(config.train, 'rb') as ftrain:
    train_input, train_global, train_label = pickle.load(ftrain)
with open(config.val, 'rb') as fval:
    val_input, val_global, val_label = pickle.load(fval)
with open(config.test, 'rb') as ftest:
    test_input, test_global, test_label = pickle.load(ftest)


# ****************************************
# hyper-parameters
learning_rate = 5e-4
batch_size = 64
sequence_length = config.max_sequence
embedding_dim = config.gc_emd_size + config.gg_emd_size
z_dim = 64
rnn_units = 128
num_flows = config.number_of_flows
verbose = 2
patience = 10
# hyper-parameters
# ****************************************

casflow_inputs = keras.layers.Input(shape=(sequence_length, embedding_dim))
bn_casflow_inputs = keras.layers.BatchNormalization()(casflow_inputs)

vae = VAE(embedding_dim, z_dim, sequence_length, rnn_units)


node_z_mean, node_z_log_var = vae.node_encoder(bn_casflow_inputs)
node_z = Sampling()((node_z_mean, node_z_log_var))
node_recon = vae.node_decode(node_z)

z_2 = tf.reshape(node_z, shape=(-1, sequence_length, z_dim))

z_mean, z_log_var = vae.encoder(z_2)
z = Sampling()((z_mean, z_log_var))


zk, logD_loss = nf_transformations(z, z_dim, num_flows)

recon_x = vae.decode(zk)


gru_1 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(rnn_units*2, return_sequences=True))(bn_casflow_inputs)
gru_2 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(rnn_units))(gru_1)

con = keras.layers.concatenate([zk, gru_2])

mlp_1 = keras.layers.Dense(128, activation='relu')(con)
mlp_2 = keras.layers.Dense(64, activation='relu')(mlp_1)
outputs = keras.layers.Dense(1)(mlp_2)

casflow = keras.Model(inputs=casflow_inputs, outputs=outputs)


optimizer = keras.optimizers.Adam(lr=learning_rate)
casflow.compile(loss=casflow_loss, optimizer=optimizer, metrics=['msle'])

train_generator = Generator(train_input, train_global, train_label, batch_size, sequence_length)
val_generator = Generator(val_input, val_global, val_label, batch_size, sequence_length)
test_generator = Generator(test_input, test_global, test_label, batch_size, sequence_length)
early_stop = keras.callbacks.EarlyStopping(monitor='val_msle', patience=patience, restore_best_weights=True)


train_history = casflow.fit_generator(train_generator,
                                    validation_data=val_generator,
                                    epochs=1000,
                                    verbose=verbose,
                                    callbacks=[early_stop],
                                    )

print('Training end!')

casflow.evaluate_generator(test_generator, verbose=1)
