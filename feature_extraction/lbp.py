# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:47:29 2021
@author: krgza
    
    Local Binary features script
    
    Usage: 
        
    file_path = .... # a string.
    
    hist_vec = lbp_hist_calculator(file_path) # returns a vector
    
"""
import cv2
import numpy as np
import pandas as pd
import os 
from tqdm import tqdm

def pixel_calculator(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except IndexError as error:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):   
    center = img[x][y]
    val_ar = []
    val_ar.append(pixel_calculator(img, center, x-1, y+1))     # top_right
    val_ar.append(pixel_calculator(img, center, x, y+1))       # right
    val_ar.append(pixel_calculator(img, center, x+1, y+1))     # bottom_right
    val_ar.append(pixel_calculator(img, center, x+1, y))       # bottom
    val_ar.append(pixel_calculator(img, center, x+1, y-1))     # bottom_left
    val_ar.append(pixel_calculator(img, center, x, y-1))       # left
    val_ar.append(pixel_calculator(img, center, x-1, y-1))     # top_left
    val_ar.append(pixel_calculator(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def lbp_hist_calculator(file_path):
    """
    Parameters
    ----------
    file_path : str
        Image file path
    Returns
    -------
    hist_lbp : Vector (256 by 1)
        Local binary features' histogram
    """
    img_bgr = cv2.imread(file_path)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])

    return hist_lbp

def hist2csv(folder,subfolder):
    """
    Parameters
    ----------
    folder : str
        Path to ..\\Caltech20.
    subfolder : str
        'training' or 'testing'
    Returns
    -------
    None.
        Create .csv file that contains features
    """
    folder_path = os.path.join(folder,subfolder)
    for category in (os.listdir(folder_path)): 
        path = os.path.join(folder_path,category)
        histograms = np.zeros((256,1,len(os.listdir(path))))
        histname = "hist_{}_{}.csv".format(category,subfolder)
        running_bar=tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path)))
        for num,img_path in running_bar:
            img_path = os.path.join(path,img_path)
            histograms[:,:,num]=np.array(lbp_hist_calculator(img_path))

            
        histograms = np.squeeze(histograms)
        df = pd.DataFrame(histograms) 
        df.to_csv(histname)
