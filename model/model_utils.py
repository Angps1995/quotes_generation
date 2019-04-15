import tensorflow as tf

def fully_connected(inputs, n_units, scope, training, activation=None):
    with tf.variable_scope(scope):
        net = tf.layers.dense(
            inputs, n_units, activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.batch_normalization(net, training=training)
        if activation is not None:
            net = activation(net)
        return net