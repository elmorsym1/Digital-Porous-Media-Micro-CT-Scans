# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:17:41 2020

@author: owner
"""
import cv2
import numpy as np
#from scipy import misc
#import PIL
#import warnings
import os
os.chdir('D:/Canada/University/PhD/Research/Programs/Python/CT data/data/')

fd = open('rawFiles/mask_dry_porespace.raw', 'rb')
rows = 601  #W=601, H=594, Slices= 1311
cols = 594
slices = 1311
f = np.fromfile(fd, dtype=np.uint8,count=rows*cols*slices)

im = f.reshape((rows, cols, slices)) #notice row, column format
fd.close()

for i in range(0,slices):
    img = im[:,:,i]
    ret2, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding
    img = 255 - img

    # save image
    cv2.imwrite("BSS_outcrop_ETC/"+str(i+1)+".png",img)