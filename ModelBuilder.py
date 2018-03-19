import tensorflow as tf
import sys


class ModelBuilder:

	def __init__(self, data):
		self.X = tf.placeholder(tf.float32, [None, data.pixels])
		self.Y = tf.placeholder(tf.float32, [None, data.classes])
		self.num_classes = data.classes
		self.keep_prob = tf.placeholder(tf.float32)
		self.isTraining = tf.placeholder(tf.bool)
		self.data  = data
		self.model = tf.reshape(self.X, shape=[-1, self.data.size, self.data.size, self.data.channels])
		self.prev_input_size = self.data.channels
		self.height = self.data.size
		self.width = self.data.size

	def flipHorizontal(self):
		# If we are currently training, perform some manipulation of the images from the input layer
		self.model = tf.cond(self.isTraining, lambda: tf.map_fn(lambda image: tf.image.random_flip_left_right(image), self.model), lambda: self.model)
		return self

	def contrast(self, lower=0.5, upper=1.5):
		self.model = tf.cond(self.isTraining, lambda: tf.minimum(tf.maximum(tf.map_fn(lambda image: tf.image.random_contrast(image, lower=lower, upper=upper), self.model), 0.0), 1.0), lambda: self.model)
		return self

	def brightness(self, upper = 0.3):
		self.model = tf.cond(self.isTraining, lambda: tf.minimum(tf.maximum(tf.map_fn(lambda image: tf.image.random_brightness(image, max_delta=upper), self.model), 0.0), 1.0), lambda: self.model)
		return self

	def saturation(self, lower=0.0, upper=2.0):
		self.model = tf.cond(self.isTraining, lambda: tf.minimum(tf.maximum(tf.map_fn(lambda image: tf.image.random_saturation(image, lower=lower, upper=upper), self.model), 0.0),1.0), lambda: self.model)
		return self

	def hue(self,  upper=0.05):
		self.model = tf.cond(self.isTraining, lambda: tf.minimum(tf.maximum(tf.map_fn(lambda image: tf.image.random_hue(image, max_delta=upper), self.model), 0.0),1.0), lambda: self.model)
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




		
