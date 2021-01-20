# The template for this code is taken from Ipek Erdogan's work given below
# https://github.com/aeytkn-tr/cmpe537-HW3/blob/main/pipelines/ipek_erdogan_pipeline.py

import os
import cv2
from scipy.spatial import distance
import numpy as np
from numpy.random import seed
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from feature_extraction.hu import hu
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier


# seed random number generator
seed(1)


def cent(k):
    centers = []
    for i in range(k):
        centers.append(randint(0, 255, 128))
    return centers


def euclidian(p1, p2):
    return distance.euclidean(p1, p2)


def Kmeans(data, k, max_iterations):
    centers = cent(k)
    for m in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for i in range(len(data)):
            distances = [euclidian((data[i]), (center)) for center in centers]
            class_ind = distances.index(min(distances))
            clusters[class_ind].append(data[i])
        for j in range(len(clusters)):
            if (len(clusters[j]) > 0):
                centers[j] = [round(sum(element) / len(clusters[j])) for element in zip(*clusters[j])]
    return centers


def getListOfFiles(dirName, labels, descriptors):
    listOfFile = os.listdir(dirName)
    # listOfFile.sort()
    hu1=hu()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            getListOfFiles(fullPath, labels, descriptors)
        elif (not (fullPath.startswith("Caltech20/training/.") or fullPath.startswith("Caltech20/testing/."))):
            label = dirName.split("/")[2]
            img = cv2.imread(fullPath)
            # resizing images to decrease the computational cost
            img = cv2.resize(img, (150, 150))
            keypoint, descriptor = hu1.detectAndCompute(img, None)
            if (descriptor is not None):  # to get rid of None's coming from sift keypoints
                descriptors.append(descriptor)
                labels.append(label)


def get_histogram(image_features, centers):
    histogram = np.zeros(len(centers))
    for f in image_features:
        idx = np.argmin(list(map(lambda c: np.linalg.norm(f - c), centers)))
        histogram[idx] += 1
    return histogram


if __name__ == '__main__':
    training_labels = []
    training_descriptors = []
    training_histograms = []
    testing_labels = []
    testing_descriptors = []
    testing_histograms = []
    #classifier = RandomForestClassifier(n_estimators=500)
    getListOfFiles("Caltech20/training/", training_labels, training_descriptors)

    getListOfFiles("Caltech20/testing/", testing_labels, testing_descriptors)
    print("Dataları sağ salim topladım.")
    data = []
    for i in range(len(training_descriptors)):
        for j in range(len(training_descriptors[i])):
            data.append(training_descriptors[i][j])

    final_centers = Kmeans(data, 20, 5)

    final_centers = np.array(final_centers)
    with open('centers.npy', 'wb') as f:
        np.save(f, final_centers)

    for k in training_descriptors:
        training_histograms.append(get_histogram(k, final_centers))

    for k in testing_descriptors:
        testing_histograms.append(get_histogram(k, final_centers))

    le = preprocessing.LabelEncoder()

    training_labels_encoded = le.fit_transform(training_labels)
    testing_labels_encoded = le.transform(testing_labels)

    class_encode_match = le.get_params()
    keys = le.classes_
    values = le.transform(le.classes_)
    dictionary = dict(zip(keys, values))

    #parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    #svc = svm.SVC()
    #classifier = GridSearchCV(svc, parameters)

    mlp_gs = MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(20, 20, 10), (20,)],
        'activation': ['relu', 'relu'],
        'learning_rate': ['constant', 'adaptive'],
    }
    classifier = GridSearchCV(mlp_gs, parameter_space)

    classifier.fit(training_histograms, training_labels_encoded)
    predictions = classifier.predict(testing_histograms)

    target_names = keys
    print(classification_report(testing_labels_encoded, predictions, zero_division=1))
    cm = confusion_matrix(testing_labels_encoded, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()