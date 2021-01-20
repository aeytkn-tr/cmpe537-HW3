# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:30:05 2021

Report File
https://docs.google.com/document/d/193kI7wIy4eNKoK9Xi64Mxd2r4drAbDCGdU5lwvRTqFw/edit#
@author: ahmet

"""

import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import cv2
import scipy
import scipy.signal as sig

class HOG:
    def __init__(self):
        self.filename=''
        self.img = cv2.imread(self.filename,0) 
        self.img = np.float32(self.img) / 255.0
        
        self.cell_size = 8
        self.n_bins = 9 # 20 degrees intervals
        self.max_val = np.max(self.oriented_grads)
        self.bucket_size = 2
    
    def get_grads(self):
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        x_grads = sig.convolve2d(self.img, kernel_x, mode='same')
        y_grads = sig.convolve2d(self.img, kernel_y, mode='same')
        
        # calculate magnitudes
        magnitudes = np.sqrt(x_grads**2+y_grads**2)
        oriented_grads = np.arctan2(x_grads,y_grads) * (180/np.pi)
        # make unsigned, scaled to 0-180degree gradients 
        for i in range(oriented_grads.shape[0]):
            for j in range(oriented_grads.shape[1]):
                oriented_grads[i,j] = [oriented_grads[i,j],360-oriented_grads[i,j]][oriented_grads[i,j]<0]
                oriented_grads[i,j] = [oriented_grads[i,j],oriented_grads[i,j]%180][oriented_grads[i,j]>180]
        # magnitudes, oriented_grads = cv2.cartToPolar(x_grads, y_grads, angleInDegrees=True)
        return(magnitudes,oriented_grads)
    
    
    def get_mag_hist_cell(self,x_loc,y_loc):
        self.hist = np.zeros(self.n_bins)
        # temp_m = self.magnitudes[320:328,450:458]
        # temp_a = self.oriented_grads[320:328,450:458]
        temp_m = self.magnitudes[x_loc:x_loc+self.cell_size,y_loc:y_loc+self.cell_size]
        temp_a = self.oriented_grads[x_loc:x_loc+self.cell_size,y_loc:y_loc+self.cell_size]    
        
        bin_vals = np.int32(np.floor(temp_a / (self.max_val/self.n_bins)))
        for x in range(self.cell_size):
            for y in range(self.cell_size):
                self.hist[bin_vals[x,y]]+=temp_m[x,y]
        return(self.hist)

    def get_magnitude_hist_block(self,loc_x, loc_y):
        # (loc_x, loc_y) defines the top left corner of the target block.
        return reduce(
            lambda arr1, arr2: np.concatenate((arr1, arr2)),
            [get_magnitude_hist_cell(x, y) for x, y in zip(
                [loc_x, loc_x + CELL_SIZE, loc_x, loc_x + CELL_SIZE],
                [loc_y, loc_y, loc_y + CELL_SIZE, loc_y + CELL_SIZE],
            )]
        )

    def forward():
        magnitudes,oriented_grads = self.get_grads()
        hist = self.get_mag_hist_cell(x_grads,y_grads)
        loc_x = loc_y = 200
        
        ydata = get_magnitude_hist_block(loc_x, loc_y)
        ydata = ydata / np.linalg.norm(ydata)
        
        xdata = range(len(ydata))
        bucket_names = np.tile(np.arange(N_BUCKETS), BLOCK_SIZE * BLOCK_SIZE)
        
        assert len(ydata) == N_BUCKETS * (BLOCK_SIZE * BLOCK_SIZE)
        assert len(bucket_names) == len(ydata)

                
# plt.imshow(oriented_grads,cmap='gray')
# plt.axis('off')
# plt.show()

# plt.imshow(magnitudes,cmap='gray')
# plt.axis('off')
# plt.show()






#res_img = np.zeros(img.shape)
# hx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
# hy = np.array([[1,2,1], [7,5,3], [-1,-2,-1]])

# for x in range(1,img.shape[0]-1):
#     for y in range(1,img.shape[1]-1):
#         res_img[x-1,y-1] = np.sum(img[x-1:x+2,y-1:y+2]*hx)
#plt.imshow(res_img,cmap='gray')
