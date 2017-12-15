import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io


def load(image_path):
	""" Loads an image from a file path

	Args:
		image_path: file path to the image

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""
	out = None
	out = io.imread(image_path)

	return out


def change_value(image):
	""" Change the value of every pixel by following x_n = 0.5*x_p^2 
		where x_n is the new value and x_p is the original value

	Args:
		image: numpy array of shape(image_height, image_width, 3)

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""

	out = None
	out = np.multiply(image, image) * 0.5

	return out


def convert_to_grey_scale(image):
	""" Change image to gray scale

	Args:
		image: numpy array of shape(image_height, image_width, 3)

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""
	out = None
	out = color.rgb2grey(image)
	# out2 = (0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]) / 255.0

	return out


def rgb_decomposition(image, channel):
	""" Return image **excluding** the rgb channel specified

	Args:
		image: numpy array of shape(image_height, image_width, 3)
		channel: str specifying the channel

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""

	out = None
	out = image.copy()
	if channel == 'R':
		out[:, :, 0] = 0
	elif channel == 'G':
		out[:, :, 1] = 0
	elif channel == 'B':
		out[:, :, 2] = 0

	return out


def lab_decomposition(image, channel):
	""" Return image decomposed to just the lab channel specified

	Args:
		image: numpy array of shape(image_height, image_width, 3)
		channel: str specifying the channel

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""

	lab = color.rgb2lab(image)
	out = None
	out = np.zeros_like(lab)
	if channel == 'L':
		out[:, :, 0] = lab[:, :, 0]
	elif channel == 'A':
		out[:, :, 1] = lab[:, :, 1]
	elif channel == 'B':
		out[:, :, 2] = lab[:, :, 2]
	out = color.lab2rgb(out)

	return out


def hsv_decomposition(image, channel='H'):
	""" Return image decomposed to just the hsv channel specified

	Args:
		image: numpy array of shape(image_height, image_width, 3)
		channel: str specifying the channel

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""

	hsv = color.rgb2hsv(image)
	out = None
	out = np.zeros_like(hsv)
	if channel == 'H':
		out[:, :, 0] = hsv[:, :, 0]
	elif channel == 'S':
		out[:, :, 1] = hsv[:, :, 1]
	elif channel == 'V':
		out[:, :, 2] = hsv[:, :, 2]
	out = color.hsv2rgb(out)

	return out


def mix_images(image1, image2, channel1, channel2):
	""" Return image which is the left of image1 and right of image 2 excluding
	the specified channels for each image

	Args:
		image1: numpy array of shape(image_height, image_width, 3)
		image2: numpy array of shape(image_height, image_width, 3)
		channel1: str specifying channel used for image1
		channel2: str specifying channel used for image2

	Returns:
		out: numpy array of shape(image_height, image_width, 3)
	"""

	out = None
	channel_to_axis = {'R': 0,
	                   'G': 1,
	                   'B': 2}
	out = np.zeros_like(image1)
	x, y, z = image1.shape
	for i in range(z):
		if i != channel_to_axis[channel1]:
			out[:, :y // 2, i] = image1[:, :y // 2, i]
		if i != channel_to_axis[channel2]:
			out[:, y // 2:, i] = image2[:, y // 2:, i]

	return out
