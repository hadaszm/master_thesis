import unittest
from river.forest.adaptive_random_forest import ARFClassifier
from semisupervised_methods.incremental_classifier import IncrementalClassifer
from constants import RANDOM_SEED
from river.datasets import synth
from unittest.mock import patch


class IncrementalClassifierTests(unittest.TestCase):

    def setUp(self):
        self.dataset = synth.Hyperplane(seed=RANDOM_SEED, n_features=2)
        self.classificators_params = {'threshold': 0.7,
                                       'classifier': ARFClassifier,
                                       'params': {'seed': RANDOM_SEED}}
        
    def init_classifiactor(self):
        return IncrementalClassifer(**self.classificators_params)

    def test_learn_one_with_label(self):
        incremental_classifier = self.init_classifiactor()
        x,y = next(self.dataset.take(1))
        self.assertEqual(incremental_classifier._timestamp,0)
        incremental_classifier = incremental_classifier.learn_one(x, y)
        self.assertEqual(incremental_classifier._timestamp,1)

    @patch('semisupervised_methods.incremental_classifier.IncrementalClassifer._check_max_probabaility')
    def test_learn_one_without_label_below_threshold(self,mock_method):
        incremental_classifier = self.init_classifiactor()
        x,y = next(self.dataset.take(1))
        self.assertEqual(incremental_classifier._timestamp,0)

        # with label 
        incremental_classifier = incremental_classifier.learn_one(x, y)
        self.assertEqual(incremental_classifier._timestamp,1)

         # without label
        mock_method.return_value = (1,0.2)
        incremental_classifier = incremental_classifier.learn_one(x)
        self.assertEqual(incremental_classifier._timestamp,1)

    @patch('semisupervised_methods.incremental_classifier.IncrementalClassifer._check_max_probabaility')
    def test_learn_one_without_label_above_threshold(self,mock_method):
        incremental_classifier = self.init_classifiactor()
        x,y = next(self.dataset.take(1))
        self.assertEqual(incremental_classifier._timestamp,0)

        # with label 
        incremental_classifier = incremental_classifier.learn_one(x, y)
        self.assertEqual(incremental_classifier._timestamp,1)

         # without label
        mock_method.return_value = (1,0.9)
        incremental_classifier = incremental_classifier.learn_one(x)
        self.assertEqual(incremental_classifier._timestamp,2)

      

if __name__ == '__main__':
    unittest.main()
