import tensorflow as tf


class Evaluator:

	def __init__(self, builder):
		self.builder = builder

		# find the prediction for current batch
		self.prediction = tf.nn.softmax(self.builder.model)

		# Count how many predictions were correct for the batch
		self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.builder.Y, 1))

		# Calculate the acuracy
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


	def softmaxCrossEntropy(self):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.builder.model, labels=self.builder.Y))

		return self

	def AdamOptimize(self):
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

		return self


		
