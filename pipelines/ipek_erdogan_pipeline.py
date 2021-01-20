###IPEK ERDOGAN COMPUTER VISION HW3 PIPELINE
###THIS PIPELINE CONSISTS OF:
###1.Compute local descriptors: OPENCV SIFT FUNCTION
###2. Find the dictionary: OWN K-MEANS IMPLEMENTATION
###3. Feature quantization: BAG OF VISUAL WORDS IMPLEMENTATION FROM MAHMUT KARACA
###4. Classification: SKLEARN RANDOM FOREST IMPLEMENTATION
###5. Evaluation

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

#####BAG OF VISUAL WORDS. THIS PART IS IMPLEMENTED BY MAHMUT KARACA#####
def get_histogram(image_features, centers):
    histogram = np.zeros(len(centers))
    for f in image_features:
        idx = np.argmin(list(map(lambda c: np.linalg.norm(f - c), centers)))
        histogram[idx] += 1
    return histogram
#######

if __name__ == '__main__':
    training_labels=[]
    training_descriptors=[]
    training_histograms = []
    testing_labels = []
    testing_descriptors = []
    testing_histograms = []
    #I AM USING RANDOM FOREST AS CLASSIFIER
    classifier = RandomForestClassifier(n_estimators = 500)

    #EXTRACTING THE DESCIPTORS FOR IMAGES AND SETTING LABEL LISTS
    descriptor_Extractor("Caltech20/training/",training_labels,training_descriptors)
    descriptor_Extractor("Caltech20/testing/", testing_labels, testing_descriptors)
    
    #MERGING ALL THE DESCRIPTORS I GOT FROM DIFFERENT IMAGES (TO GIVE IT TO K MEANS)
    data=[]
    for i in range(len(training_descriptors)):
        for j in range(len(training_descriptors[i])):
            data.append(training_descriptors[i][j])

    #KMEANS
    final_centers = Kmeans(np.array(data), 100, 10)

    #SAVING THE CENTERS TO USE THEM LATER
    final_centers=np.array(final_centers)
    with open('centers.npy', 'wb') as f:
        np.save(f, final_centers)

  
    #CREATING TRAINING AND TESTING HISTOGRAMS(FEATURE VECTORS) BY USING BAG OF VISUAL WORDS IMPLEMENTATION OF MAHMUT
    for k in training_descriptors:
        training_histograms.append(get_histogram(k,final_centers))
    for k in testing_descriptors:
        testing_histograms.append(get_histogram(k,final_centers))
    
    #LABEL ENCODING
    le = preprocessing.LabelEncoder()
    training_labels_encoded = le.fit_transform(training_labels)
    testing_labels_encoded = le.transform(testing_labels)
    keys = le.classes_
    values = le.transform(le.classes_)
    dictionary = dict(zip(keys, values))

    #TRAINING RANDOM FOREST AND RUNING IT ON TEST DATA TO MAKE PREDICTIONS
    classifier.fit(training_histograms, training_labels_encoded)
    predictions = classifier.predict(testing_histograms)
    target_names = ['Faces','airplanes','anchor','barrel','camera','car_side','dalmatian','ferry','headphone','lamp','pizza','pyramid','snoopy','soccer_ball','stop_sign','strawberry','sunflower','water_lilly','windsor_chair','yin_yang']
    
    #EVALUATION PART WHICH CONSISTS OF MEAN F1-SCORE, PER-CLASS (F1-SCORE, PRECISION, RECALL) AND CONFUSION MATRIX
    print(classification_report(testing_labels_encoded, predictions, target_names=target_names,zero_division=1))
    cm = confusion_matrix(testing_labels_encoded, predictions, labels=values)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = keys)
    disp.plot()
    plt.show()

