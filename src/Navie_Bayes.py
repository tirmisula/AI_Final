import numpy as np
import math
from collections import Counter


class NavieBayes(object):
    def __init__(self, feature_matrix, labels):
        self.total_labels = len(labels)
        self.labeled_features = {}
        for feature, label in zip(feature_matrix, labels):
            if label not in self.labeled_features:
                self.labeled_features[label] = feature
            else:
                self.labeled_features[label] = np.concatenate(self.labeled_features[label], feature, axis=0)
        self.labeled_mean = {}
        self.labeled_standard_deviation = {}
        self.label_counts = Counter(labels)

    def train(self):
        for label, features in self.labeled_features.items():
            self.labeled_mean[label] = np.mean(features, axis=0)
            self.labeled_standard_deviation[label] = np.std(features, axis=0)

    def predict(self, feature):
        prob_list = []
        for ele in self.labeled_features.keys():
            normal = normal_wrapper(self.labeled_mean[ele], self.labeled_standard_deviation[ele])
            label_prob = self.label_counts(ele) / self.total_labels
            prob_list.append(calculate_probablitiy(feature, normal) * label_prob)

        return np.argmax(prob_list)


def normal_distribution(x, mean, standard_deviation):
    square_deviation = float(standard_deviation) ** 2
    pi = math.pi
    denom = (2 * pi * square_deviation) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * square_deviation))
    return num / denom


def normal_wrapper(mean, standard_deviation):
    def normal(x, index):
        return normal_distribution(x, mean[index], standard_deviation[index])
    return normal


def calculate_probablitiy(feature, probablity_function):
    limit = len(feature)
    probablity = 1
    for index in range(limit):
        probablity *= probablity_function(feature[index], index)

    return probablity_function

