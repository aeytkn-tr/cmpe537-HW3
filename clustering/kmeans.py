###IPEK ERDOGAN COMPUTER VISION HW3 K MEANS IMPLEMENTATION
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

# seed random number generator
seed(1)

##########THIS PART IS FOR CLUSTERING. MY OWN IMPLEMENTATION OF K-MEANS.
def cent(k,nr_el=128):
    centers=[]
    for i in range(k):
        centers.append(randint(0, 255, nr_el))
    return centers

def euclidian(p1,p2):
    return distance.euclidean(p1, p2)

def Kmeans(data,k,max_iterations):
    centers=cent(k,data.shape[1])
    for m in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for i in range(len(data)):
            distances=[euclidian((data[i]),(center)) for center in centers]
            class_ind = distances.index(min(distances))
            clusters[class_ind].append(data[i])
        for j in range(len(clusters)):
            if(len(clusters[j])>0):
                centers[j] = [round(sum(element) / len(clusters[j])) for element in zip(*clusters[j])]
    return centers
#################################