import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from PIL import Image

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
	    shuffled_data = [self.trainX[i] for i in idx]
	    shuffled_labels = [self.trainY[i] for i in idx]

	    return np.asarray(shuffled_data), np.asarray(shuffled_labels)

	def next_test_batch(self, num):	

	    idx = np.arange(0 , len(self.testX))
	    np.random.shuffle(idx)
	    idx = idx[:num]
	    shuffled_data = [self.testX[ i] for i in idx]
	    shuffled_labels = [self.testY[ i] for i in idx]

	    return np.asarray(shuffled_data), np.asarray(shuffled_labels)

	def greyscale(self):

		if self.channels == False:
			print('Error. Must be RGB to convert to greyscale.')
			sys.exit(1)

		sess = tf.InteractiveSession()
		self.formatToTensor()

		self.channels = 1
		self.pixels = self.size * self.size

		self.trainX = tf.image.rgb_to_grayscale(self.trainX)
		self.testX = tf.image.rgb_to_grayscale(self.testX)

		self.formatToFlat()


		return self
	def formatToTensor(self):
		self.trainX = tf.reshape(self.trainX, shape=[-1, self.size, self.size, self.channels])
		self.testX = tf.reshape(self.testX, shape=[-1, self.size, self.size, self.channels])

	def formatToFlat(self):
		self.trainX = tf.reshape(self.trainX, [-1, self.size * self.size * self.channels]).eval()
		self.testX = tf.reshape(self.testX, [-1, self.size * self.size * self.channels]).eval()

	def toOneHot(x):
		if x == 0:
			return [1,0]
		else:
			return [0,1]

	def get_data(folder='Combined'):
	    path1 = folder  # path of folder of images
	    listing = os.listdir(path1)
	    num_samples=np.size(listing)


	    imlist = os.listdir(path1)
	    im1 = cv2.imread(folder +'/'+ imlist[0], cv2.IMREAD_COLOR)
	    m = np.size(im1, 0)
	    n = np.size(im1, 1)  # get the size of the images
	    imnbr = len(imlist)  # get the number of images

	    # create matrix to store all flattened images
	    immatrix = np.array([np.array(cv2.imread(folder + '/' + im2, cv2.IMREAD_COLOR)).flatten()
	                      for im2 in imlist], 'f')
	    label = np.ones((num_samples,), dtype=int)
	    label[0:303] = 0
	    label[304:636] = 1
	    data, Label = shuffle(immatrix, label, random_state=2)
	    train_data = [data, Label]

	    # Divide into data and labels
	    (X, Y) = (train_data[0], train_data[1])

	    # STEP 1: split X and y into training and testing sets

	    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)



	    # Scale the values from 0 to 1 instead of 0 to 255
	    X_train = X_train.astype('float32')
	    X_test = X_test.astype('float32')
	    # Return the sets to their original shapes
	    X_train = X_train.reshape(-1, 50, 50, 3)
	    X_test = X_test.reshape(-1, 50, 50, 3)
	    X_train /= 255
	    X_test /= 255

	    y_train = list(map(lambda x: Data.toOneHot(x), y_train))
	    y_test = list(map(lambda x: Data.toOneHot(x), y_test))


	    return (X_train, y_train), (X_test, y_test)
