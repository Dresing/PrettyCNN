import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from PIL import Image

class Manipulator:

	def __init__(self):
		self.images = None

	def

	def applyOn(model):
		return tf.map_fn(lambda image: self.flipHorizontal(image), images)

	def flipHorizontal(self, image):
		return tf.image.random_flip_left_right(image)