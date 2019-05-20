import tensorflow as tf

def model(vocab_size=9320, embedding_dim=100, batch_size=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, 100]),
        tf.keras.layers.GRU(256,
                            return_sequences=False,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size,activation='softmax')
      ])
    return model

if __name__ == "__main__":
    model = model()
    print(model.summary())