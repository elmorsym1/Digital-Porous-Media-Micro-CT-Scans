# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:17:41 2020

@author: owner
"""
import cv2
import numpy as np
import pandas as pd
import os
import subprocess


# Dataname
dataname = "Beadpack_150_1"

# Image number for points in mesh
img_name = "150.png"

img_path = "data/"+dataname+"/"+img_name


num_points = 1
#print(num_points_in)


# sample resolution
res = 1

save_dir = "newcase/"
cont_ex_f = 0.1
os.path.splitext(img_name)



X = os.path.splitext(img_name)[0] 

# Read image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding
img_inv = 255 - img

# calculate porosity [void_ratio]
img_h = img.shape[0]
img_w = img.shape[1]
void_area= np.sum(img_inv/255)
img_porosity = void_area/(img_h*img_w)
#if img_porosity < 0.5:
#    img_p = img_inv
#else:
#    img_p = img

img_p = img_inv 


# define circles
radius = img_h // 6
yc = img_h // 2
xc = img_w // 2

# draw filled circle in white on black background as mask
mask = np.zeros_like(img_p)
mask = cv2.circle(mask, (xc,yc), radius, (255,255,255), -1)

# apply mask to image
img_masked = cv2.bitwise_and(img_p, mask)


# Calculating contours
contours, hierarchy = cv2.findContours(img_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cont_ex = cont_ex_f*min(img_w,img_h)

# select one contour to porocess
cont=[]  
for i in range(len(contours)):
    if len(contours[i])>cont_ex:
        cont.append(contours[i])

max_num_points = len(cont)
print("Image used for extracting points in mesh is "+str(img_name))        
print("Max. number of points in Mesh = "+str(max_num_points))
print("Exported number of points to snappyhexMesh = "+str(min(num_points, max_num_points)))


df_headerfile =  pd.DataFrame({"//Header file for points in Mesh"})

if max_num_points < 1:
  
  pt_val = {"pointsInMesh unvalid;"}
  print("There are no pointsInMesh to extract!")
  df_pt_mesh = pd.concat([df_headerfile, pd.DataFrame(pt_val)], axis=0, sort=False)
  df_pt_mesh.to_csv(save_dir+"pointsInMesh.h", index=False, header=False)
  
else:

  pt_val = {"pointsInMesh valid;"}
  print("pointsInMesh are extracted successfully!")
        
  # define matrices = []
  area = []
  pt_meshX = []
  pt_meshY= []
  pt_meshZ= []
  
  # Creat white image
  img_cont = np.ones((img_p.shape[0],img_p.shape[1],3))*255
  img_pt = np.ones((img_p.shape[0],img_p.shape[1],3))*255
  
  # Calculate area for each contour
  for i in range(len(cont)):
      # creating convex hull object for each contour
      area.append(cv2.contourArea(cont[i]))
  
  # Sort contours based on area
  sort_index = np.argsort(area)[::-1] # [::-1] Invert it to "Decending order"
  
  for i in sort_index:
      mask = np.zeros(img_p.shape,np.uint8) # Creat black image
      cv2.drawContours(mask,cont,i,255,-1)  # Mask each contour area on it 
      pt_list = np.squeeze(cv2.findNonZero(mask), axis=1) # List of each contour area pixels
      
      # loop to find the farthest point in mesh for each contour
      pt_dist_m = []
      for pt in pt_list:
          pt_dist = cv2.pointPolygonTest(cont[i], (pt[0], pt[1])  , True)
          pt_dist_m.append(pt_dist)
          
      c_pt_index = pt_dist_m.index(max(pt_dist_m))
      pt_meshX.append(X)
      pt_meshY.append(pt_list[c_pt_index,0])
      pt_meshZ.append(pt_list[c_pt_index,1])
  
  # draw the contour and point in mesh of the contour on the image
  for i in range(len(cont)):
      cv2.drawContours(img_pt, cont[i], -1, (0,0,255), 1)
      cv2.circle(img_pt, (pt_meshY[i], pt_meshZ[i]), 2, (0,0,0), -1)
      
  # save image
  cv2.imwrite(save_dir+"pointsInMesh.png",img_pt)
  
  # convert data to dataframes
  df_head = pd.DataFrame({'c1': ["#Points"],'c2': [len(pt_meshX)]})
  df_pt_meshX = pd.DataFrame({'c1':pt_meshX})
  df_pt_meshY = pd.DataFrame({'c2':pt_meshY})
  df_pt_meshZ = pd.DataFrame({'c3':pt_meshZ})
  df_points_combined = pd.concat([df_pt_meshX, df_pt_meshY, df_pt_meshZ], axis=1, sort=False) # axis = 1 [X direction, add as columns]
  df_points_data = pd.concat([df_head, df_points_combined], axis=0, sort=False) # axis = 0 [Y direction, add as rows]
  
  ## save data to csv file
  df_points_data.to_csv(save_dir+"pointsInMesh.csv", index=False, header=False)  
  
  
  # save header file for points in mesh
  pt_x = []
  pt_y = []
  pt_z = []
  
  for p in range(0,num_points):
       i = str(p+1)
       if p>max_num_points-1:
          p=max_num_points-1
       x = str(float(pt_meshX[p])*res)
       y = str(float(pt_meshY[p])*res)
       z = str(float(pt_meshZ[p])*res)
  
       pt_x.append("x"+i+" "+x+";")
       pt_y.append("y"+i+" "+y+";")
       pt_z.append("z"+i+" "+z+";")
  
  
    
  df_pt_mesh = pd.concat([df_headerfile, pd.DataFrame(pt_x), pd.DataFrame(pt_y), pd.DataFrame(pt_z), pd.DataFrame(pt_val)], axis=0, sort=False)
  df_pt_mesh.to_csv(save_dir+"pointsInMesh.h", index=False, header=False)
  
  