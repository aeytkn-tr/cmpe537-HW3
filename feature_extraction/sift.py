###IPEK ERDOGAN COMPUTER VISION HW3 SIFT PART

import os
import cv2
from scipy.spatial import distance
import numpy as np
from numpy.random import seed
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#########IN THIS PART, I AM USING PREPROCESSING AND USING OPENCV SIFT IMPLEMENTATION TO EXTRACT THE DESCRIPTORS
def descriptor_Extractor(dirName,labels,descriptors):
    listOfFile = os.listdir(dirName)
    #listOfFile.sort()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if descriptor_Extractor(fullPath,labels,descriptors)
        elif(not (fullPath.startswith("Caltech20/training/.") or fullPath.startswith("Caltech20/testing/."))):
            label = dirName.split("/")[2]
            img = cv2.imread(fullPath)
            #resizing images to decrease the computational cost
            img = cv2.resize(img,(150, 150))
            #convert images to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #create SIFT object
            sift = cv2.xfeatures2d.SIFT_create(100)
            #detect SIFT features in both images
            keypoint, descriptor = sift.detectAndCompute(img, None)
            if (len(keypoint) >= 1): #to get rid of None's coming from sift keypoints
                descriptors.append(descriptor)
                labels.append(label)
##################################