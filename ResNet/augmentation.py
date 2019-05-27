import cv2
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as functional

class HorizontalFlip(object):
	def __call__(self, img):
		if random.random() < 0.5:
			img = np.copy(np.fliplr(img))
		return img

class VerticalFlip(object):
	def __call__(self, img):
		if random.random() < 0.5:
			img = np.copy(np.flipud(img))
		return img

class ColorWarp(object):
	def __init__(self, mean_range=0, std_range=0):
		super(ColorWarp, self).__init__()
		self.mean_range = mean_range
		self.std_range = std_range

	def __call__(self, img):
		if random.random() < 0.5:
			std = np.random.uniform(-self.std_range, self.std_range, 3)
			mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
			order = np.random.permutation(3)

			img *= (1 + std)
			img += mean
			img = img[:, :, order]

		return img

class GaussianIllumination(object):
	def __init__(self, mean, std):
		super(GaussianIllumination, self).__init__()
		self.mean = mean
		self.std = std

	def __call__(self, img):
		if random.random() < 0.5:
			additive = np.random.normal(self.mean, self.std, 1)
			img = np.clip(img + additive, 0, 1).astype(np.float32)
		return img

class ContrastAdjust(object):
	def __init__(self, low, high):
		super(ContrastAdjust, self).__init__()
		self.low = low
		self.high = high

	def __call__(self, img):
		contrast_factor = np.random.uniform(self.low, self.high)
		if not isinstance(img, Image.Image):
			img = np.clip(img*255, 0, 255).astype(np.uint8)
			img = Image.fromarray(img)
		img = functional.adjust_contrast(img, contrast_factor)
		return img

class GammaAdjust(object):
	def __init__(self, low, high):
		super(GammaAdjust, self).__init__()
		self.low = low
		self.high = high

	def __call__(self, img):
		gamma = np.random.uniform(self.low, self.high)
		img = functional.adjust_gamma(img, gamma)
		return img

class BrightnessAdjust(object):
	def __init__(self, mean, std):
		super(BrightnessAdjust, self).__init__()\
		self.mean = mean
		self.std = std

	def __call__(self, img):
		brightness = np.random.normal(self.mean, self.std)
		img = functional.adjust_brightness(img, 1+brightness)
		return img

class SaturationAdjust(object):
	def __init__(self, low, high):
		super(SaturationAdjust, self).__init__()
		self.low = low
		self.high = high

	def __call__(self, img):
		saturation = np.random.uniform(self.low, self.high)
		img = functional.adjust_saturation(img, saturation)
		return img

class HueAdjust(object):
	def __init__(self, low, high):
		super(HueAdjust, self).__init__()
		self.low = low
		self.high = high

	def __call__(self, img):
		hue = np.random.uniform(self.low, self.high)
		img = functional.adjust_hue(img, hue)
		return img

class RandomScale(object):
	def __init__(self, low, high):
		super(RandomScale, self).__init__()
		self.low = low
		self.high = high

	def __call__(self, img):
		scale_factor = np.random.uniform(self.low, self.high)
		img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

class CenterCrop(object):
	def __init__(self, crop_size):
		super(CenterCrop, self).__init__()
		self.th, self.tw = crop_size

	def __call__(self, img):
		h, w = img.size
		img = img[(h-self.th)//2:(h+self.th)//2, (w-self.tw)//2:(w+self.tw)//2, :]
		return img
