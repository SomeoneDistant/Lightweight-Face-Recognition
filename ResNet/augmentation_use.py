import augmentation as aug
#import dataloader
import cv2
import numpy as np
from PIL import Image


augmentations = aug.Sequential([
	aug.HorizontalFlip(),
	aug.ColorWarp(mean_range=0.2, std_range=0.2),
	aug.GaussianIllumination(mean=0.2,std=0.1),
	aug.RandomScale(low=0.98,high=1),
	aug.CenterCrop(crop_size=[110,110]),
	aug.RandomRotate(low=-2,high=2),
	aug.RandomNoise(),
	aug.ContrastAdjust(low=0.3,high=10),
	aug.GammaAdjust(low=0.3,high=2.5),
	aug.BrightnessAdjust(mean=1,std=0.1),
	aug.SaturationAdjust(low=-0.5,high=5)
	])

#format h,w,c read from opencv directly 
def augImg(img):
	img = img / 255   
	img=np.array(img).astype(np.float32)
	img=augmentations(img)
	img = np.array(img)
	if img.shape[0]!=112:
		img=cv2.resize(img,(112,112))
	return img






