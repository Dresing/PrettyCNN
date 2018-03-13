import tensorflow as tf
import numpy as np

class Data:

	def __init__(self, trainX, trainY, testX, testY, img_size, channels):
		self.trainX = trainX
		self.trainY = trainY
		self.testX = testX
		self.testY = testY

		self.pixels = len(self.trainX[0])
		self.classes = len(self.trainY[0])
		self.size = img_size
		self.channels = channels

	def next_train_batch(self, num):

	    idx = np.arange(0 , len(self.trainX))
	    np.random.shuffle(idx)
	    idx = idx[:num]
	    shuffled_data = [self.trainX[ i] for i in idx]
	    shuffled_labels = [self.trainY[ i] for i in idx]

	    return np.asarray(shuffled_data), np.asarray(shuffled_labels)

	def next_test_batch(self, num):

	    idx = np.arange(0 , len(self.testX))
	    np.random.shuffle(idx)
	    idx = idx[:num]
	    shuffled_data = [self.testX[ i] for i in idx]
	    shuffled_labels = [self.testY[ i] for i in idx]

	    return np.asarray(shuffled_data), np.asarray(shuffled_labels)

