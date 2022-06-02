import pickle
import time

from absl import app, flags

from utils.tools import *

# flags
FLAGS = flags.FLAGS
flags.DEFINE_float  ('lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('b_size', 64, 'Batch size.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('emb_dim', 40+40, 'Embedding dimension (cascade emb_dim + global emb_dim')
flags.DEFINE_integer('z_dim', 64, 'Dimension of latent variable z.')
flags.DEFINE_integer('rnn_units', 128, 'Number of RNN units.')
flags.DEFINE_integer('n_flows', 8, 'Number of NF transformations.')
flags.DEFINE_integer('verbose', 2, 'Verbose.')
flags.DEFINE_integer('patience', 10, 'Early stopping patience.')

# paths
flags.DEFINE_string ('input', './dataset/sample/', 'Dataset path.')


def main(argv):

    start_time = time.time()
    print('TF Version:', tf.__version__)

    with open(FLAGS.input + 'train.pkl', 'rb') as ftrain:
        train_cascade, train_global, train_label = pickle.load(ftrain)
    with open(FLAGS.input + 'val.pkl', 'rb') as fval:
        val_cascade, val_global, val_label = pickle.load(fval)
    with open(FLAGS.input + 'test.pkl', 'rb') as ftest:
        test_cascade, test_global, test_label = pickle.load(ftest)

    casflow_inputs = tf.keras.layers.Input(shape=(FLAGS.max_seq, FLAGS.emb_dim))
    bn_casflow_inputs = tf.keras.layers.BatchNormalization()(casflow_inputs)

    # node-level uncertainty
    node_emb = tf.keras.layers.Dense(FLAGS.emb_dim)(bn_casflow_inputs)
    node_mean = tf.keras.layers.Dense(FLAGS.z_dim)(node_emb)
    node_log_var = tf.keras.layers.Dense(FLAGS.z_dim)(node_emb)
    node_z = Sampling3D()((node_mean, node_log_var))

    node_rec = tf.keras.layers.Dense(FLAGS.z_dim)(node_z)
    node_rec = tf.keras.layers.Dense(FLAGS.emb_dim)(node_rec)

    # cascade-level uncertainty
    cas_emb = tf.keras.layers.GRU(FLAGS.rnn_units)(node_z)
    cas_mean = tf.keras.layers.Dense(FLAGS.z_dim)(cas_emb)
    cas_log_var = tf.keras.layers.Dense(FLAGS.z_dim)(cas_emb)
    cas_z = Sampling2D()((cas_mean, cas_log_var))

    # normalizing transformations
    zk, logD_loss = nf_transformations(cas_z, FLAGS.z_dim, FLAGS.n_flows)

    # reconstruct node_z
    cas_recon = tf.keras.layers.RepeatVector(FLAGS.max_seq)(zk)
    cas_recon = tf.keras.layers.GRU(FLAGS.rnn_units, return_sequences=True)(cas_recon)
    cas_recon = tf.keras.layers.Dense(FLAGS.z_dim)(cas_recon)

    gru_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(FLAGS.rnn_units * 2, return_sequences=True))(bn_casflow_inputs)
    gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(FLAGS.rnn_units))(gru_1)

    con = tf.keras.layers.Concatenate()([zk, gru_2])

    mlp_1 = tf.keras.layers.Dense(128, activation='relu')(con)
    mlp_2 = tf.keras.layers.Dense(64, activation='relu')(mlp_1)
    outputs = tf.keras.layers.Dense(1)(mlp_2)

    casflow = tf.keras.Model(inputs=casflow_inputs, outputs=outputs)

    # cal node-level vae losses
    node_ce_loss = tf.reduce_mean(tf.square(bn_casflow_inputs - node_rec))
    node_kl_loss = -.5 * tf.reduce_mean(node_log_var - tf.square(node_mean) - tf.exp(node_log_var) + 1)
    casflow.add_loss(node_ce_loss)
    casflow.add_loss(node_kl_loss)
    # cal cascade-level vae losses
    cas_ce_loss = tf.reduce_mean(tf.square(node_z - cas_recon))
    cas_kl_loss = -.5 * tf.reduce_mean(cas_log_var - tf.square(cas_mean) - tf.exp(cas_log_var) + 1)
    casflow.add_loss(cas_ce_loss)
    casflow.add_loss(cas_kl_loss)
    casflow.add_loss(-.1 * tf.reduce_mean(logD_loss))

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    casflow.compile(loss='msle', optimizer=optimizer, metrics=['msle'])

    train_generator = Generator(train_cascade, train_global, train_label, FLAGS.b_size, FLAGS.max_seq)
    val_generator = Generator(val_cascade, val_global, val_label, FLAGS.b_size, FLAGS.max_seq, is_train=False)
    test_generator = Generator(test_cascade, test_global, test_label, FLAGS.b_size, FLAGS.max_seq, is_train=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_msle', patience=FLAGS.patience, restore_best_weights=True)

    casflow.fit(train_generator,
                validation_data=val_generator,
                epochs=1000,
                verbose=FLAGS.verbose,
                callbacks=[early_stop],)

    print('Training ended!')

    # metrics MSLE and MAPE reported in paper are defined as follows
    # (note the base of the logarithm is 2)
    predictions = np.array([1 if pred < 1 else pred for pred in np.squeeze(casflow.predict(test_generator))])
    test_label = np.array([1 if label < 1 else label for label in test_label])
    report_msle = np.mean(np.square(np.log2(predictions) - np.log2(test_label)))
    report_mape = np.mean(np.abs(np.log2(predictions + 1) - np.log2(test_label + 1)) / np.log2(test_label + 2))

    print('Test MSLE: {:.4f}, MAPE: {:.4f}'.format(report_msle, report_mape))

    print('Finished! Time used: {:.3f}mins.'.format((time.time()-start_time)/60))


if __name__ == '__main__':
    app.run(main)