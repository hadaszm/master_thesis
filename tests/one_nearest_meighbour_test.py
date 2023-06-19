import unittest
import pandas as pd
import numpy as np
from semisupervised_methods.one_nearest_neighbour import oneNNClassifer
from constants import RANDOM_SEED
from river.datasets import synth


class oneNNClassiferTests(unittest.TestCase):

    
    def setUp(self):
        self.dataset = synth.Hyperplane(seed=RANDOM_SEED, n_features=2)

    def test_learn_one(self):
        clf = oneNNClassifer(training_time=2,positive=1,negative=0, required_ratio=0.5)
        x1 = dict(zip([1,2,3],[1, 2, 3]))
        x2 = dict(zip([1,2,3],[4,5,6]))
        y1 = 1
        y2 = 0

        clf.learn_one(x1, y1)

        self.assertEqual(len(clf.L), 1)
        self.assertEqual(len(clf.U), 0)
        self.assertEqual(clf._timestamp, 1)

        clf.learn_one(x2, y2)

        self.assertEqual(len(clf.L), 1)
        self.assertEqual(len(clf.U), 1)
        self.assertEqual(clf._timestamp, 1)

        clf.learn_one(x2, y1)
        self.assertEqual(clf._timestamp, 2)
       

    def test_predict_one(self):
        clf = oneNNClassifer(training_time=2,positive=1,negative=0, required_ratio=0.5)
        x1 = dict(zip([1,2,3],[1, 2, 3]))
        x2 = dict(zip([1,2,3],[4, 5, 6]))
        y1 = 1
        y2 = 0

        clf.learn_one(x1, y1)
        clf.learn_one(x2, y2)
        clf.learn_one(x1, y1)


        prediction1 = clf.predict_one(x1)
        prediction2 = clf.predict_one(x2)

        self.assertEqual(prediction1, y1)
        self.assertEqual(prediction2, y2)


if __name__ == '__main__':
    unittest.main()
