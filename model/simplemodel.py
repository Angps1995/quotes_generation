import tensorflow as tf

def simplemodel(vocab_size=9955,max_len=816):
    inp = tf.keras.layers.Input(shape=(max_len,),name='input')
    net = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=100,input_length=max_len)(inp)
    net = tf.keras.layers.CuDNNGRU(256)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(256,activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(vocab_size,activation='softmax',name='output')(net)
    model = tf.keras.models.Model(inputs=inp,outputs=net)
    return model

if __name__ == "__main__":
    model = simplemodel()
    print(model.summary())