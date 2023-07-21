import albumentations as alb
import matplotlib.pyplot as plt
# from albumentations.pytorch import ToTensor
import os, shutil
import cv2
import numpy as np
import random

from scipy import ndimage
# from skimage.feature import peak_local_max
# from skimage.morphology import watershed
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage import io
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

def augment(image, masks):

    transform = alb.Compose([
        alb.RandomRotate90(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        # alb.OneOf([

        #   alb.augmentations.transforms.CLAHE(clip_limit=3),          
        #   alb.augmentations.transforms.GaussNoise(var_limit=(20.0)),
        #   alb.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        #   alb.augmentations.transforms.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
        #   alb.augmentations.transforms.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
        #   alb.augmentations.transforms.GaussNoise(var_limit=(2.5500000000000003, 12.75), per_channel=False, p=0.5)
        # ], p= 0.6),

        # alb.OneOf([
        #   alb.Blur(blur_limit=3),
        #   alb.GaussianBlur(blur_limit=3, sigma_limit=0, always_apply=False, p=0.5),
        #   alb.MedianBlur(blur_limit=3, always_apply=False, p=0.5),
        # ]),

        alb.augmentations.geometric.transforms.Affine(scale=1.0, translate_percent=0, translate_px=None, rotate=(-90, 90), shear=0.0, interpolation=1, cval=0),

    ], p=1)

    transformed = transform(image=image, masks=masks)
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']

    return transformed_image, transformed_masks

def pre_process(mask):

    def Gauss2D(X, Y, sigma, a, b):
        Z = 0.06 * np.exp(-((X - a) ** 2 + (Y - b) ** 2) / sigma ** 2)
        return Z


    def Heatsum(A1, A2):
        B = 1 - (1 - A1) * (1 - A2)
        return B
    
    h, w = mask.shape
    re = h
    sigma = 1.6
    x = np.arange(0, re, 1)
    y = np.arange(0, re, 1)
    X, Y = np.meshgrid(x, y)

    # mask = cv2.threshold(mask, 126, 255, cv2.THRESH_BINARY)[1]/ 255.

    contour00 = cv2.Canny(mask, 0, 1)
    contour00 = cv2.normalize(contour00, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    contour00 = contour00.astype(np.double)

    y, x = np.where(contour00 != 0)
    Z = np.zeros_like(X)
    Zsum = np.zeros_like(Z)

    for j in range(len(y)):
        Z = Gauss2D(X, Y, sigma, x[j], y[j])
        Zsum = Heatsum(Zsum, Z)

    contour = (Zsum - np.min(Zsum)) / (np.max(Zsum) - np.min(Zsum))
    contour[contour < 0.001] = 0

    dis = distance_transform_edt(1.0 - mask)
    dis = cv2.normalize(dis, None, 0., 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    s_dis_1 = -(distance_transform_edt(1.0 - contour00) - distance_transform_edt(1.0 - mask))
    s_dis = cv2.normalize(s_dis_1, None, -1.0, 0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    s_dis11 = dis + s_dis

    return contour, s_dis11

