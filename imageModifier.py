# To gain diversity of image in limited sample amount, generate variation of given image 


import numpy as np
import cv2
import os
import random

def addSaltPepperToImg(img):
    saltRatio = 0.006
    pepperRatio = 0.001
    
    tainted_img = np.copy(img)
    num_salt = np.ceil(img.size*saltRatio)
    coords = tuple([np.random.randint(0,i-1,int(num_salt)) for i in img.shape])
    tainted_img[coords] = 1
    
    num_pepper = np.ceil(img.size*pepperRatio)
    coords = tuple([np.random.randint(0,i-1,int(num_pepper)) for i in img.shape])
    tainted_img[coords] = 0
    
    return tainted_img

def addGaussianNoiseToImg(img):
    row,col,ch = img.shape
    mean = 0
    var = 81
    sigma = var**0.5
    
    gaussian = np.random.normal(mean, sigma, (row,col,ch))
    noisy_img = img + gaussian
    
    cv2.normalize(noisy_img, noisy_img,0,255,cv2.NORM_MINMAX, dtype=-1)
    noisy_img = noisy_img.astype(np.uint8)
    
    return noisy_img

def addNoise(img):
    resImgObj = None
    dice = random.randint(1,4)
    if dice % 4 == 0:
        resImgObj = addGaussianNoiseToImg(img)
    elif dice % 4 == 1:
        resImgObj = addSaltPepperToImg(img)
    else:
        resImgObj = addSaltPepperToImg(img)
        resImgObj = addGaussianNoiseToImg(resImgObj)
    
    return resImgObj 
