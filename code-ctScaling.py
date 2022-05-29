#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import os
from os import walk
import math
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

os.chdir('D:/Canada/University/PhD/Research/Programs/Python/CT data')
# Data directory path
data_name = "BrSS_cube400_ICL_test"
original_resolution = 5.345 #um
target_resolution = 3 #um

data_path = "data/"+data_name+"/"
data_savename = data_name+"_nR_"+str(target_resolution)+"um"
save_path = "data/"+data_savename+"/"


for (dirpath, dirnames, filenames) in walk(data_path):
    break
# extract images extension and their number
img_name, img_ext = os.path.splitext(data_path+filenames[0])
num_imgs = len(filenames)

##########################################################################
##########################################################################
# custome_functions
   
def porosity_cube(cube):
  cube_h = cube.shape[0]
  cube_w = cube.shape[1]
  cube_d = cube.shape[2]
  # ret2, cube = cv2.threshold(cube,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding
  cube = 255 - cube
  void_vol= np.sum(cube/255)
  porosity_cube = void_vol/(cube_h*cube_w*cube_d)
  return porosity_cube

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


# calculate porosity [void_ratio] for 3D volume
original_size_px = img_3d.shape[0]
original_size_um = original_size_px*original_resolution
original_porosity = porosity_cube(img_3d)


summary_file_path = 'data/'+data_savename+'_summary.txt'
with open(summary_file_path, 'w') as f:
    with redirect_stdout(f):
        print("original size [um] = "+str(original_size_um))
        print("original size [px] = "+str(original_size_px))
        print("original Res. [um/px] = "+str(original_resolution))
        print("original porosity = "+str(original_porosity))


##########################################################################
##########################################################################
# reScale CT voulme
new_size_px = np.floor(original_size_um/target_resolution)
factor = new_size_px/original_size_px
img_3d_rs = zoom(img_3d, (factor, factor, factor))


# Otsu's thresholding after Gaussian filtering
for i in range(0, img_3d_rs.shape[2]):
    img_3d_rs[:, :, i] = cv2.GaussianBlur(img_3d_rs[:, :, i],(5,5),0)
    ret2, img_3d_rs[:, :, i] = cv2.threshold(img_3d_rs[:, :, i],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding


new_size_px = img_3d_rs.shape[0]
new_resolution = original_size_um/new_size_px
new_porosity = porosity_cube(img_3d_rs)
dp = ((new_porosity-original_porosity)/original_porosity)

with open(summary_file_path , 'a') as f:
    with redirect_stdout(f):
        print("new size [um] = "+str(new_size_px*new_resolution))
        print("new size [px] = "+str(new_size_px))
        print("new Res. [um/px] = "+str(new_resolution))
        print("new porosity = "+str(new_porosity))
        print("Porosity dif. (%) = "+str(dp*100))

f = open(summary_file_path , 'r')
print(f.read())        

# Save New CT Cube
os.mkdir(save_path)
for n in range(0,img_3d_rs.shape[0]):
  img_save = img_3d_rs[:,:,n]
  cv2.imwrite(save_path+str(n+1)+".png",img_save)


