import numpy as np


class Perceptron(object):
    def __init__(self, classes=None, epoch=20, error_rate=0.05):
        # Each feature gets its own weight vector, so weights is a dict-of-arrays
        self.classes = classes
        self.weights = []
        self.epoch = epoch
        self.error = 0
        self.threshold = error_rate

    def calculate_score(self, feature):
        result = np.dot(self.weights, feature)
        label_index = np.argmax(result)
        output_array = np.zeros(self.classes)
        output_array[label_index] = 1
        return output_array, label_index

    def predict(self, feature):
        _, index = self.calculate_score(self, feature)
        return index

    def train(self, features: np.ndarray, labels):
        def initialize_weights(features):
            dimension = features.shape[0]
            self.weights = np.random.random_sample((self.classes, dimension + 1))
            self.weights[:, -1] = np.ones(dimension)

        initialize_weights(features)
        biases = np.random.random_sample((features.shape[0], 1))
        features = np.concatenate(features, biases, axis=1)
        error_rate = 0
        for num in range(1, self.epoch + 1):
            for index in range(features):
                feature = features[index]
                predict, predict_index = self.calculate_score(feature)
                if predict_index != labels[index]:
                    self.weights[labels[index]] += feature
                    self.weights[predict_index] -= feature
                    self.error += 1
            error_rate = (error_rate + self.error / features.shape[0]) / num
            self.error = 0
            if error_rate < self.threshold:
                break
