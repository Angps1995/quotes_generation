import tensorflow as tf
from model.model import model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = '/home/angps/Documents/Quotes_generation/model_logs/model.hdf5')
    parser.add_argument('--export_path', default = '/home/angps/Documents/Quotes_generation/model_logs/export/1')
    args = parser.parse_args()
    model = model()
    model.load_weights(args.model)

    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            args.export_path,
            inputs={'input': model.input},
            outputs={'output': model.output})