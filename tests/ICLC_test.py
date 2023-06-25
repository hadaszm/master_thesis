import unittest
from river.forest.adaptive_random_forest import ARFClassifier
from river.cluster  import KMeans
from river.drift import ADWIN
from semisupervised_methods.ICLC import ICLC
from constants import RANDOM_SEED
from river.datasets import synth
import unittest.mock as mock
from unittest.mock import patch


class ICLCTest(unittest.TestCase):

    def setUp(self):
        self.dataset = synth.Hyperplane(seed=RANDOM_SEED, n_features=2)
        self.classifier_params = {}
        self.clustering_params = {"n_clusters": 2, "seed": RANDOM_SEED}
        self.nu = 10
        self.drift_detector = ADWIN

    def _initialize(self):
        return ICLC(
            classifer=ARFClassifier,
            clustering_method=KMeans,
            classifier_params=self.classifier_params,
            clustering_params=self.clustering_params,
            nu=self.nu,
            drift_detector=self.drift_detector
        )

    def test_initialize_ICLC(self):
        iclc = self._initialize()

        self.assertIsInstance(iclc, ICLC)
        self.assertIsInstance(iclc.classifier, ARFClassifier)
        self.assertIsInstance(iclc.clustering_method, KMeans)
        self.assertEqual(iclc.nu, self.nu)
        self.assertEqual(iclc.drift_detector, self.drift_detector)

    @patch('semisupervised_methods.ICLC.ICLC._check_if_drift_detetcted')
    def test_learn_one_labeled_instance_drift(self,drift_detected):
        iclc = self._initialize()

        # no drift
        drift_detected.return_value = False
        iclc = iclc.learn_one(next(self.dataset.take(1))[0])
        self.assertNotAlmostEquals(len(iclc.clustering_method.centers[0].values()),0)
        self.assertEqual(iclc._unlabelled_instances_cnt,1)

        # drift
        drift_detected.return_value = True
        iclc = iclc.learn_one(next(self.dataset.take(1))[0])
        self.assertEqual(len(iclc.clustering_method.centers[0].values()),0)
        self.assertEqual(iclc._unlabelled_instances_cnt,0)


        self.assertIsInstance(iclc.classifier, ARFClassifier)

    def test_learn_one_labeled_instance(self):
        iclc = self._initialize()

        x,y = next(self.dataset.take(1))

        iclc = iclc.learn_one(x,y)

        self.assertEqual(iclc._unlabelled_instances_cnt,0)
        self.assertEqual(iclc._timestamp, 1)
        self.assertIsInstance(iclc.classifier, ARFClassifier)
        self.assertIsInstance(iclc.clustering_method, KMeans)

    def test_learn_one_unlabeled_instance(self):
        iclc = self._initialize()

        x,y = next(self.dataset.take(1))

        iclc = iclc.learn_one(x)

        self.assertEqual(iclc._unlabelled_instances_cnt,1)
        self.assertEqual(iclc._timestamp, 0)
        self.assertIsInstance(iclc.classifier, ARFClassifier)
        self.assertIsInstance(iclc.clustering_method, KMeans)

    def test_predict_one(self):
        iclc = self._initialize()
        for x,y in self.dataset.take(10):
            iclc = iclc.learn_one(x,y)

        x,y = next(self.dataset.take(1))
        prediction = iclc.predict_one(x)
        self.assertIsInstance(prediction, type(y))

    @patch('semisupervised_methods.ICLC.ICLC._learn_from_unlabelled')
    def test_train_on_unlabelled_called(self,mock):
        iclc = self._initialize()
        iclc._unlabelled_instances_cnt = 10

        x,y = next(self.dataset.take(1))
        iclc = iclc.learn_one(x,y)
        self.assertTrue(mock.called)

    def test_learn_from_unlabelld(self):
        iclc = self._initialize()
        for x,y in self.dataset.take(10):
            iclc = iclc.learn_one(x,y)
        for x,y in self.dataset.take(10):
            iclc = iclc.learn_one(x)

        x,y = next(self.dataset.take(1))
        iclc = iclc.learn_one(x,y)
        self.assertEqual(iclc._timestamp, 13)  





    if __name__ == '__main__':
        unittest.main()


