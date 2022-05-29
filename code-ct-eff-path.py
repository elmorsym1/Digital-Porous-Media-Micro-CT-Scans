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
import cc3d
import matplotlib.pyplot as plt

os.chdir('D:/Canada/University/PhD/Research/Programs/Python/CT data')
# Data directory path
data_name = "BrSS_cube400_ICL_test"
original_resolution = 5.345 #um
target_resolution = 3 #um

data_path = "data/"+data_name+"/"
data_savename = data_name+"_nR_"+str(target_resolution)+"um"
save_path = "data/"+data_savename+"/"
os.mkdir(save_path)

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
print("original size [um] = "+str(original_size_um))
print("original porosity = "+str(original_porosity))

##########################################################################
##########################################################################
# reScale CT voulme
new_size_px = np.floor(original_size_um/target_resolution)
factor = new_size_px/original_size_px
img_3d_rs = zoom(img_3d, (factor, factor, factor), mode = 'constant', grid_mode=True)
for i in range(0, img_3d_rs.shape[2]):
    ret2, img_3d_rs[:, :, i] = cv2.threshold(img_3d_rs[:, :, i],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding

new_resolution = original_size_um/img_3d_rs.shape[0]
new_porosity = porosity_cube(img_3d_rs)
dp = ((new_porosity-original_porosity)/original_porosity)
print("new resolution = "+str(new_resolution))
print("new porosity = "+str(new_porosity))
print("Porosity dif. (%) = "+str(dp*100))
# Save New CT Cube
for n in range(0,img_3d_rs.shape[0]):
  img_save = img_3d_rs[:,:,n]
  cv2.imwrite(save_path+str(n+1)+".png",img_save)

##########################################################################
##########################################################################
# Document Porosity
"""
df_OIS = img_3d.shape[0]
df_NIS = img_3d_rs.shape[0]
df_OP = pd.DataFrame({'Original Porosity': original_porosity})
df_NP = pd.DataFrame({'New Porosity': new_porosity})
df_DP = pd.DataFrame({'Dif. %': dp})
df_OR = pd.DataFrame({'Original Resulotion [um]': original_resolution})
df_NR = pd.DataFrame({'New Resulotion [um]': new_resolution})

df_info = pd.concat([df_OIS, df_NIS, df_OP, df_NP, df_DP, df_OR, df_NR], axis=1, sort=False) # X dir                             
## save data to csv file
df_info.to_csv(save_path+data_savename+"_porosity.csv", index=False, header=True)
"""
##########################################################################
##########################################################################
im = img_3d[:, :, 0]
structure = np.zeros((3, 3), dtype=np.int)  # this defines the connection filter
labeled, ncomponents = label(im, structure)
plt.imshow(labeled)
##########################################################################
##########################################################################
# VIP can identify the continous sapce 
##########################################################################
##########################################################################
# https://pypi.org/project/connected-components-3d/
img_3d_inv = 255 - img_3d
labels_in = img_3d_inv
labels_out, N = cc3d.connected_components(labels_in, return_N=True)

unique, counts = np.unique(labels_out, return_counts=True)
eff_porosity = counts[1]/np.sum(counts)
print("eff. porosity = "+str(eff_porosity))
labels_out_1D = labels_out.reshape(-1)
plt.hist(labels_out_1D, bins=N)
# labels_out_new = labels_out.astype("uint8")
labels_out[labels_out > 1] = 0
labels_out = labels_out.astype("uint8") #continous space
plt.hist(labels_out.reshape(-1), bins=255)

plt.imshow(labels_in[:, :, 0])
plt.imshow(labels_out[:, :, 0])

# plt.imshow(labels_out_new[:, :, 0])

edges = cc3d.region_graph(labels_out, connectivity=6)
# Specify connected direcions
graph = cc3d.voxel_connectivity_graph(labels_out, connectivity=6)

##########################################################################
##########################################################################
# Label all connected components:
im = img_3d[:, :, 0]
im =  255 - im
all_labels = measure.label(im)
plt.imshow(all_labels)

# Label only foreground connected components
blobs_labels = measure.label(im, background=0)
plt.imshow(blobs_labels)

# 3D case
all_labels_3D = measure.label(img_3d)
plt.imshow(all_labels_3D[:, :, 0])

blobs_labels_3D = measure.label(img_3d, background=0)
plt.imshow(blobs_labels_3D[:, :, 0])


##########################################################################
##########################################################################
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html
# Distance Map, try different distanceType
# cv2.DIST_L2, the simple euclidean distance
img_inv = 255 - im
distmap = cv2.distanceTransform(img_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
plt.imshow(im)
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
