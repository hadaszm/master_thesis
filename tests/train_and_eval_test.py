import unittest
from train_and_eval import update_performance_measures
from river.metrics.accuracy import Accuracy

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

if __name__ == '__main__':
    unittest.main()
