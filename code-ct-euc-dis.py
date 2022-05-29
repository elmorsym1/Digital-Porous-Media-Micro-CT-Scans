#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import os
from os import walk
import math
from skimage import measure
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.chdir('D:/Canada/University/PhD/Research/Programs/Python/CT data')
# Data directory path
data_name = "BrSS_cube400_ICL_test"

data_path = "data/"+data_name+"/"
data_savename = data_name+"_ecu_dist"
save_path = "data/"+data_savename+"/"

for (dirpath, dirnames, filenames) in walk(data_path):
    break
# extract images extension and their number
img_name, img_ext = os.path.splitext(data_path+filenames[0])
num_imgs = len(filenames)

##########################################################################
##########################################################################
# loop over images to build 3d voulme
for i in range(1,num_imgs+1):
    img_num = str(i)
    img_path = data_path+img_num+img_ext
    if i == 1:  # load first image
        img_3d = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h = img_3d.shape[0]
        w = img_3d.shape[1]
        ret2, img_3d = cv2.threshold(img_3d,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding
    else:       # Build 3d array
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ret2, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding
        img_3d = np.dstack((img_3d, img))


im  = img_3d[:, :, 0]
plt.imshow(im, cmap=cm.Greys_r)

##########################################################################
##########################################################################
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html
# Distance Map, try different distanceType
# cv2.DIST_L2, the simple euclidean distance
im  = img_3d[:, :, 0]
img_inv = 255 - im
distmap = cv2.distanceTransform(img_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
plt.imshow(im, cmap=cm.Greys_r)
plt.imshow(distmap)

# Distance Map, try different distanceType
# cv2.DIST_L1, distance = |x1-x2| + |y1-y2|, Try this first
img_inv = 255 - im
distmap = cv2.distanceTransform(img_inv, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
plt.imshow(distmap)

# Distance Map, try different distanceType
# cv2.DIST_LABEL_PIXEL, result is identical to L1
img_inv = 255 - im
distmap = cv2.distanceTransform(img_inv, cv2.DIST_LABEL_PIXEL, cv2.DIST_MASK_PRECISE)
plt.imshow(distmap)


# Distance Map, try different distanceType
# cv2.DIST_C
img_inv = 255 - im
distmap = cv2.distanceTransform(img_inv, cv2.DIST_C, cv2.DIST_MASK_PRECISE)
plt.imshow(distmap)

# Distance Map, try different distanceType
# cv2.DIST_MASK_3,  result is identical to C
img_inv = 255 - im
distmap = cv2.distanceTransform(img_inv, cv2.DIST_MASK_3, cv2.DIST_MASK_PRECISE)
plt.imshow(distmap)



# Distance Map 3D
distmap_3D = cv2.distanceTransform(img_3d, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
plt.imshow(distmap_3D[:, :, 0])
