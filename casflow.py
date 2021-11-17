import pickle
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
flags.DEFINE_string ('input', './dataset/xovee/', 'Dataset path.')


def main(argv):

    start_time = time.time()
    print('TF Version:', tf.__version__)

    def casflow_loss(y_true, y_pred):
        mse = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
        node_ce_loss_x = tf.reduce_mean(
            keras.losses.mean_squared_error(bn_casflow_inputs, node_recon))
        node_kl_loss = - 0.5 * tf.reduce_mean(
            1 + node_z_log_var - tf.square(node_z_mean) - tf.exp(node_z_log_var))

        ce_loss_x = tf.reduce_mean(
            keras.losses.mean_squared_error(bn_casflow_inputs, recon_x))
        kl_loss = - 0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        return mse + node_ce_loss_x + ce_loss_x + node_kl_loss + kl_loss - logD_loss

    with open(FLAGS.input + 'train.pkl', 'rb') as ftrain:
        train_cascade, train_global, train_label = pickle.load(ftrain)
    with open(FLAGS.input + 'val.pkl', 'rb') as fval:
        val_cascade, val_global, val_label = pickle.load(fval)
    with open(FLAGS.input + 'test.pkl', 'rb') as ftest:
        test_cascade, test_global, test_label = pickle.load(ftest)

    casflow_inputs = keras.layers.Input(shape=(FLAGS.max_seq, FLAGS.emb_dim))
    bn_casflow_inputs = keras.layers.BatchNormalization()(casflow_inputs)

    vae = VAE(FLAGS.emb_dim, FLAGS.z_dim, FLAGS.max_seq, FLAGS.rnn_units)

    node_z_mean, node_z_log_var = vae.node_encoder(bn_casflow_inputs)
    node_z = Sampling()((node_z_mean, node_z_log_var))
    node_recon = vae.node_decode(node_z)

    z_2 = tf.reshape(node_z, shape=(-1, FLAGS.max_seq, FLAGS.z_dim))

    z_mean, z_log_var = vae.encoder(z_2)
    z = Sampling()((z_mean, z_log_var))

    zk, logD_loss = nf_transformations(z, FLAGS.z_dim, FLAGS.n_flows)

    recon_x = vae.decode(zk)

    gru_1 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(FLAGS.rnn_units * 2, return_sequences=True))(
        bn_casflow_inputs)
    gru_2 = keras.layers.Bidirectional(keras.layers.CuDNNGRU(FLAGS.rnn_units))(gru_1)

    con = keras.layers.concatenate([zk, gru_2])

    mlp_1 = keras.layers.Dense(128, activation='relu')(con)
    mlp_2 = keras.layers.Dense(64, activation='relu')(mlp_1)
    outputs = keras.layers.Dense(1)(mlp_2)

    casflow = keras.Model(inputs=casflow_inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(lr=FLAGS.lr)
    casflow.compile(loss=casflow_loss, optimizer=optimizer, metrics=['msle'])

    train_generator = Generator(train_cascade, train_global, train_label, FLAGS.b_size, FLAGS.max_seq)
    val_generator = Generator(val_cascade, val_global, val_label, FLAGS.b_size, FLAGS.max_seq)
    test_generator = Generator(test_cascade, test_global, test_label, FLAGS.b_size, FLAGS.max_seq)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_msle', patience=FLAGS.patience, restore_best_weights=True)

    train_history = casflow.fit_generator(train_generator,
                                          validation_data=val_generator,
                                          epochs=1000,
                                          verbose=FLAGS.verbose,
                                          callbacks=[early_stop])

    print('Training ended!')

    casflow.evaluate_generator(test_generator, verbose=1)

    predictions = [1 if pred < 1 else pred for pred in np.squeeze(casflow.predict_generator(test_generator))]
    test_label = [1 if label < 1 else label for label in test_label]

    # metrics MSLE and MAPE reported in paper are defined as
    report_msle = np.mean(np.square(np.log2(predictions) - np.log2(test_label)))
    report_mape = np.mean(np.abs(np.log2(np.array(predictions) + 1) - np.log2(np.array(test_label) + 1))
                          / np.log2(np.array(test_label) + 2))

    print('Test MSLE: {:.4f}, MAPE: {:.4f}'.format(report_msle, report_mape))

    print('Finished! Time used: {:.3f}mins.'.format((time.time()-start_time)/60))


if __name__ == '__main__':
    app.run(main)
