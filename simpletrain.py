import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model.simplemodel import simplemodel

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, required=True)
	parser.add_argument('--direc', default="/home/angps/Documents/Quotes_generation/data/")
	parser.add_argument('--batch_size', type=int, default=128)
	args = parser.parse_args()

	direc = args.direc
	def data_gen(typ):
		maxlen = 816
		vocab_size = 9955
		fl = np.load(direc + typ + '_quotes.npy')
		while True:
			num = np.random.choice(len(fl)-1)
			quotes = fl[num]
			idx = np.random.choice(len(quotes))
			in_quotes, out_quotes = quotes[:idx], quotes[idx]
			in_quotes = pad_sequences([in_quotes],maxlen = maxlen).flatten()  
			out_quotes = to_categorical(out_quotes,num_classes = vocab_size)
			yield np.expand_dims(in_quotes,axis=0), np.expand_dims(out_quotes,axis=0)

	optimizer = tf.keras.optimizers.Adam()

	train_gen = data_gen('train')
	test_gen = data_gen('val')

	checkpoint_path = "/home/angps/Documents/Quotes_generation/model_logs/model.hdf5"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	model = simplemodel()
	model.compile(optimizer='adam',
				loss='categorical_crossentropy')
	try:			  
		#latest = tf.train.latest_checkpoint(checkpoint_dir)
		model.load_weights(checkpoint_path)
	except:
		print('')
		#model.load_weights(checkpoint_path)
	#model.load_weights(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
													save_weights_only=True,
													verbose=1) 
	tb = tf.keras.callbacks.TensorBoard(log_dir='/home/angps/Documents/Quotes_generation/model_logs',  
			write_graph=True, write_images=True)
	learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
												patience=3, 
												verbose=1, 
												factor=0.1, 
												min_lr=0.00001)
	history = model.fit_generator(train_gen, 
						steps_per_epoch=50,
						validation_data=test_gen,
						validation_steps=25,
						epochs=args.epochs,
						callbacks=[cp_callback,tb],
						verbose=1)