import tensorflow as tf
import numpy as np
from os import path
import h5py
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from simplemodel import simple

def model_fn(features, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    output = simple(features, training=training)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'output': output
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.keras.losses.CategoricalCrossentropy
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
        maxlen = 817
        vocab_size = 9955
        fl = np.load(file)
        while True:
            num = np.random.choice(len(fl)-1)
            quotes = fl[num]
            idx = np.random.choice(len(quotes)-2) + 1
            in_quotes, out_quotes = quotes[:idx], quotes[idx]
            in_quotes = pad_sequences([in_quotes],maxlen = maxlen).flatten()  
            out_quotes = to_categorical(out_quotes,num_classes = vocab_size)
            yield (in_quotes,out_quotes)

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