
import csv
import random
import math
import operator
import cv2
import pandas as pd

# toán khoảng cách Euclid:
def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


# Tự tìm k láng giềng gần nhất:
def kNearestNeighbors(training_feature_vector, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance,
                training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Tự bỏ phiếu quyết định nhãn:
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# Load image feature data to training feature vectors and test feature vector
def loadDataset(training_filename, test_filename, training_feature_vector=[], test_feature_vector=[]):
    training_data = pd.read_csv(training_filename, header=None)
    for index, row in training_data.iterrows():
        # Lấy tất cả các cột, bao gồm cả nhãn màu ở cột cuối
        training_feature_vector.append(row.values.tolist())

    test_data = pd.read_csv(test_filename, header=None)
    for index, row in test_data.iterrows():
        test_feature_vector.append(row.values.tolist())

def main(training_data, test_data):
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    loadDataset(training_data, test_data, training_feature_vector, test_feature_vector)
    classifier_prediction = []  # predictions
    k = 7  # K value of k nearest neighbor
    for x in range(len(test_feature_vector)):
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x], k)
        result = responseOfNeighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]
