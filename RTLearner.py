# this is moddified from my own code project 3: RTLearner.py


import numpy as np
from scipy import stats

class RTLearner(object):


    def author(self):
        return "gwang383"

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def build_tree(self, dataset, leaf_size):
        size = dataset.shape[0]
        feature_size = dataset.shape[1] - 1
        unique_count = np.unique(dataset[:, -1]).shape[0]
        if unique_count == 1:
            return np.array([[-1, dataset[0, -1], -1, -1]])

        if size <= leaf_size:
            result = stats.mode(dataset[:, -1])[0]
            return np.array([[-1, result, -1, -1]])

        feature = np.random.randint(low=0, high=feature_size)
        value = np.median(dataset[:, feature])

        if dataset[dataset[:, feature] <= value].shape[0] == size:
            result = stats.mode(dataset[:, -1])[0]
            return np.array([[-1, result, -1, -1]])


        left_tree = self.build_tree(dataset[dataset[:, feature] <= value], leaf_size)
        right_tree = self.build_tree(dataset[dataset[:, feature] > value], leaf_size)
        root = np.array([feature, value, 1, left_tree.shape[0] + 1])
        return np.vstack((root, left_tree, right_tree))


    def add_evidence(self, data_x, data_y):
        feature_size = data_x.shape[1]
        data = np.zeros((data_x.shape[0], feature_size + 1))
        data[:, 0:feature_size] = data_x
        data[:, feature_size] = data_y
        self.tree = self.build_tree(data, leaf_size=self.leaf_size)

    def query(self, points):
        size = points.shape[0]
        exp = np.zeros(size)
        for j in range(0, size):
            start = 0
            for k in range(0, self.tree.shape[0]):
                feature, value, left, right = self.tree[start]
                feature, left, right = int(feature), int(left), int(right)
                if feature == -1:
                    exp[j] = value
                    break
                if points[j, feature] > value:
                    p = right
                else:
                    p = left
                start += p
        return exp
