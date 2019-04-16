import tensorflow as tf

def simplemodel(vocab_size=9955,max_len=816):
    inp = tf.keras.layers.Input(shape=(max_len,))
    net = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=256,input_length=max_len)(inp)
    net = tf.keras.layers.LSTM(512)(net)
    net = tf.keras.layers.Dense(1024,activation='relu')(net)
    net = tf.keras.layers.Dense(vocab_size,activation='softmax')(net)
    model = tf.keras.models.Model(inputs=inp,outputs=net)
    return model

if __name__ == "__main__":
    model = simplemodel()
    print(model.summary())