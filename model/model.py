import tensorflow as tf
import numpy as np
from os import path
import h5py
from encoders import encoder, encoder_v2,encoder_v3,encoder_v4
from decoders import decoder_folding, decoder_fc


def model_fn(features, mode, params):


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'code_word': code_word,
            'output': reconstruction
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    loss_forward, _, loss_backward, _ = \
        tf_nndistance.nn_distance(inputs, reconstruction)

    loss = tf.reduce_mean(loss_forward) + tf.reduce_mean(loss_backward)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN
    learning_rate = tf.train.exponential_decay(
        params['lr'], tf.train.get_global_step(),
        decay_rate=params['lr_decay_rate'],
        decay_steps=params['lr_decay_steps'],
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss,
                                      train_op=train_op)


class Generator:
    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            for sample in hf['data']:
                yield sample

def input_fn(data_dir, data_filenames, input_shape,
             batch_size=1, shuffle=True, n_epochs=None):
    file_paths = [path.join(data_dir, f) for f in data_filenames]
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.interleave(lambda filepath: tf.data.Dataset.from_generator(
        Generator(),
        tf.float32,
        tf.TensorShape(input_shape),
        args=(filepath,)), cycle_length=1, block_length=5000)

    ds = ds.batch(5000).repeat(n_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds



def train_input_fn(data_dir, hyperparameters):
    return input_fn(
        data_dir,
        input_shape=hyperparameters['input_shape'],
        data_filenames=hyperparameters['train_data_filenames'],
        batch_size=hyperparameters['batch_size'],
        shuffle=True,
        n_epochs=hyperparameters['epochs'])


def eval_input_fn(data_dir, hyperparameters=None):
    return input_fn(
        data_dir,
        input_shape=hyperparameters['input_shape'],
        data_filenames=hyperparameters['val_data_filenames'],
        batch_size=hyperparameters['batch_size'],
        n_epochs=1,
        shuffle=False)


def predict_input_fn(data_dir, hyperparameters):
    return input_fn(
        data_dir,
        input_shape=hyperparameters['input_shape'],
        data_filenames=hyperparameters['predict_data_filenames'],
        batch_size=hyperparameters['batch_size'],
        n_epochs=1,
        shuffle=False)


#def serving_input_fn(hyperparameters):
def serving_input_fn():
    #points = tf.placeholder(tf.float32, shape=hyperparameters['input_shape'])
    points = tf.placeholder(tf.float32, shape=(None,500,4))
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        {'points': points})()