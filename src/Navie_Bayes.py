import numpy as np


class NavieBayes(object):
    def __init__(self, feature_matrix, labels):
        self.features = feature_matrix
        self.means = []
        self.variances = []
        self.labels = labels

    def train(self):
        self.means = np.mean(self.features, axis=0)

