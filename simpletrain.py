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
	def data_gen(typ,batch_size):
		maxlen = 816
		vocab_size = 9955
		fl = np.load(direc + '/data/' + typ + '_quotes.npy')
		while True:
			in_quotes = []
			out_quotes = []
			for i in range(batch_size):
				num = np.random.choice(len(fl)-1)
				quotes = fl[num]
				idx = np.random.choice(len(quotes))
				in_quote, out_quote = quotes[:idx], quotes[idx]
				in_quote = pad_sequences([in_quote],maxlen = maxlen).flatten()  
				out_quote = to_categorical(out_quote,num_classes = vocab_size)
				in_quotes = in_quotes + [in_quote] 
				out_quotes = out_quotes + [out_quote] 
			yield np.reshape(in_quotes,(batch_size,maxlen)),np.reshape(out_quotes,(batch_size,vocab_size))

	optimizer = tf.keras.optimizers.Adam()
	base_lr = 0.001
	train_gen = data_gen('train',args.batch_size)
	test_gen = data_gen('val',args.batch_size)

	checkpoint_path = "/home/angps/Documents/Quotes_generation/model_logs/model.hdf5"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	model = simplemodel()
	model.compile(optimizer='adam',
				loss='categorical_crossentropy')
	#model.load_weights(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
													save_weights_only=True,
													verbose=1) 
	tb = tf.keras.callbacks.TensorBoard(log_dir='/home/angps/Documents/Quotes_generation/model_logs/./Graph',  
			write_graph=True, write_images=True)
	learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
												patience=3, 
												verbose=1, 
												factor=0.1, 
												min_lr=0.00001)
	def lr_linear_decay(epoch):
    	return (base_lr * (1 - (epoch / num_epochs)))

	history = model.fit_generator(train_gen, 
						steps_per_epoch=5000,
						validation_data=test_gen,
						validation_steps=2500,
						epochs=num_epochs,
						callbacks=[tf.keras.callbacks.LearningRateScheduler(
										lr_linear_decay), cp_callback,tb],
						verbose=1)