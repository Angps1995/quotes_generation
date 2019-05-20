import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model.model import model
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--direc', default="/home/angps/Documents/Quotes_generation/data/")
	parser.add_argument('--batch_size', type=int, default=128)
	args = parser.parse_args()

	DIREC = args.direc
	BATCH = args.batch_size
	EPOCHS = args.epochs

	def data_gen(typ,batch_size):
		maxlen = 100
		vocab_size = 9320
		fl = np.load(DIREC + typ + '_quotes_maxlen100.npy',allow_pickle=True)
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

	model = model()
	model.compile(optimizer='adam', loss='categorical_crossentropy')

	optimizer = tf.keras.optimizers.Adam()
	base_lr = 0.001
	train_gen = data_gen('train',BATCH)
	test_gen = data_gen('val',BATCH)

	checkpoint_path = DIREC + "model_logs/model.hdf5"
	checkpoint_dir = os.path.dirname(checkpoint_path)


	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
													save_weights_only=True,
													save_best_only=True,
													verbose=1) 
	tb = tf.keras.callbacks.TensorBoard(log_dir=DIREC+'model_logs/./Graph',  
			write_graph=True, write_images=True)
	learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
												patience=3, 
												verbose=1, 
												factor=0.1, 
												min_lr=0.00001)
	def lr_linear_decay(epoch):
		return (base_lr * (1 - (epoch / EPOCHS)))
				

	history = model.fit_generator(train_gen, 
						steps_per_epoch=3500,
						validation_data=test_gen,
						validation_steps=3500,
						epochs=EPOCHS,
						callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_linear_decay), cp_callback,tb],verbose=1)