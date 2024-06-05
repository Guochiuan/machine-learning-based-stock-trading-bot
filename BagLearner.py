# this is moddified from my own code project 3: BagLearner.py

import numpy as np
import RTLearner as rt
from scipy import stats


class BagLearner(object):
    def __init__(self, learner=rt.RTLearner, kwargs={}, bags=10, boost=False, verbose=False):
        self.bags = bags
        self.boost = boost
        self.learner = learner
        self.verbose = verbose
        self.learners = []
        for i in range(self.bags):
            self.learners.append(self.learner(**kwargs))

    def author(self):
        return "gwang383"

    def add_evidence(self, data_x, data_y):
        sample_size = data_x.shape[0]
        feature_size = data_x.shape[1]
        dataset = np.zeros([sample_size, feature_size + 1])
        dataset[:, 0: -1] = data_x
        dataset[:, -1] = data_y
        for i in range(0, self.bags):
            temp = np.zeros([sample_size, feature_size + 1])
            random = np.random.randint(0, sample_size, sample_size)
            for j in range(0, sample_size):
                temp[j] = dataset[random[j]]
            random_x = temp[:, 0:-1]
            random_y = temp[:, -1]
            self.learners[i].add_evidence(random_x, random_y)

    def query(self, points):
        # print(points)
        size = points.shape[0]
        result = np.zeros((self.bags, size))
        for i, learner in enumerate(self.learners):
            result[i] = learner.query(points)


        # print(stats.mode(result, axis=0)[0])
        return stats.mode(result, axis=0)[0]
