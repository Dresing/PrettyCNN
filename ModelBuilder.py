import tensorflow as tf
import sys


class ModelBuilder:

	def __init__(self, data, model):
		self.X = tf.placeholder(tf.float32, [None, data.pixels])
		self.Y = tf.placeholder(tf.float32, [None, data.classes])
		self.num_classes = data.classes
		self.keep_prob = tf.placeholder(tf.float32)
		self.isTraining = tf.placeholder(tf.bool)
		self.data  = data
		self.manipulator = manipulator
		self.model = model




	def input(self):

		self.model = tf.reshape(self.X, shape=[-1, self.data.size, self.data.size, self.data.channels])


		self.prev_input_size = self.data.channels
		self.height = self.data.size
		self.width = self.data.size

		# If we are currently training, perform some manipulation of the images from the input layer
		if self.manipulator is not None:
			self.model = tf.cond(self.isTraining, lambda: self.manipulator.applyOn(self.model), lambda: self.model)


		return self


	def conv(self, filters = 32, size = 5, strides = 1):

		## Add convolutional weight: size, size, input, output (filters)
		W = tf.Variable(tf.random_normal([size, size, self.prev_input_size, filters]))

		## Construct convolutional layer
		self.model = tf.nn.conv2d(self.model, W, strides=[1, strides, strides, 1], padding='SAME')

		## Define bias based on number of of output filters
		b = tf.Variable(tf.random_normal([filters]))

		self.model = tf.nn.bias_add(self.model, b)

		## Aply relu for non-linearity
		self.model = tf.nn.relu(self.model)

		## Set input size for next layer
		self.prev_input_size = filters

		return self

	def pool(self, size = 2, strides = 2):

		## Perform max pooling layer
		self.model = tf.nn.max_pool(self.model, ksize=[1, size, size, 1], strides=[1, strides, strides, 1], padding='SAME')

		## Recalculate model size
		self.width = int((self.width - size) / strides + 1) 
		self.height = int((self.height - size) / strides + 1) 

		return self


	def reshape(self):
		self.width += 1
		self.height += 1

		self.model = tf.reshape(self.model, [-1, self.width * self.height * self.prev_input_size])

		return self

	def dense(self, units=1024):
		W = tf.Variable(tf.random_normal([self.width * self.height * self.prev_input_size, units]))
		b = tf.Variable(tf.random_normal([units]))
		self.model = tf.add(tf.matmul(self.model, W), b)
		self.model = tf.nn.relu(self.model)

		self.prev_input_size = units
		return self

	def dropout(self):
		self.model = tf.nn.dropout(self.model, self.keep_prob)
		return self

	def logits(self):
		W = tf.Variable(tf.random_normal([self.prev_input_size, self.num_classes]))
		b = tf.Variable(tf.random_normal([self.num_classes]))
		self.model = tf.add(tf.matmul(self.model, W), b)
		return self




		
