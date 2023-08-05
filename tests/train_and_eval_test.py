import unittest
from train_and_eval import update_performance_measures
from river.metrics.accuracy import Accuracy
import numpy as np

class TestTrainAndEval(unittest.TestCase):
    def test_update_performance_measures(self):
        # Create a sample input
        predictions = [{ 1: 1,2: 0, 3: 0, 4: 1, 5: 0},
                       { 1: 1,2: 1, 3: 0, 4: 0}]

        true_label = 1
        B = 2
        metrics = [Accuracy() for _ in range(B+2)]

        
        for preds in predictions:
            update_performance_measures(preds, true_label, B, metrics)

        self.assertEqual([t.get() for t in metrics],[1, 0.5, 0.5, 0])


    def test_update_performance_measures2(self):
        # Create a sample input
        predictions = [dict(zip(range(5),[1 for _ in range(5)]))]

        true_label = 1
        B = 50
        metrics = [Accuracy() for _ in range(B+2)]

        
        for preds in predictions:
            update_performance_measures(preds, true_label, B, metrics)

        pass

        self.assertEqual([t.get() for t in metrics],[1 for _ in range(50)])


if __name__ == '__main__':
    unittest.main()
