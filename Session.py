import tensorflow as tf


class Session:

	def __init__(self, session, evaluator):
		self.evaluator = evaluator
		self.init = tf.global_variables_initializer()
		self.sess = session
		self.sess.run(self.init)
		self.data = evaluator.builder.data
		self.learning_rate = 0.001
		self.num_steps = 50
		self.batch_size = 128
		self.display_step = 10
		self.keep_prob = 0.7


	def steps(self, num):
		self.num_steps = num
		return self

	def rate(self, rate):
		self.learning_rate = rate
		return self

	def batch(self, size):
		self.batch_size = size
		return self

	def dropout(self, rate):
		self.keep_prob = (1.0-rate)
		return self

	def statusEvery(self, steps):
		self.display_step = steps
		return self


	def train(self):

	    for step in range(1, self.num_steps+1):
	        batch_x, batch_y = self.data.next_train_batch(self.batch_size)
	        # Run optimization op (backprop)
	        self.sess.run(self.evaluator.optimizer, feed_dict={self.evaluator.builder.X: batch_x, self.evaluator.builder.Y: batch_y, self.evaluator.builder.keep_prob: 0.75})
	        if step % self.display_step == 0 or step == 1:
	            # Calculate batch loss and accuracy
	            loss, acc = self.sess.run([self.evaluator.loss, self.evaluator.accuracy], feed_dict={self.evaluator.builder.X: batch_x,
	                                                                 self.evaluator.builder.Y: batch_y,
	                                                                 self.evaluator.builder.keep_prob: 1.0})
	            print("Step " + str(step) + ", Minibatch Loss= " + \
	                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.3f}".format(acc))

	    print("Optimization Finished!")

	    return self
	def test(self):
		print("Testing Accuracy:", \
			self.sess.run(self.evaluator.accuracy, feed_dict={self.evaluator.builder.X: self.data.testX[:256],
		                                      self.evaluator.builder.Y: self.data.testY[:256],
		                                      self.evaluator.builder.keep_prob: 1.0}))
		return self