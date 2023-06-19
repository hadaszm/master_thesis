import unittest
from collections import Counter, defaultdict
from constants import RANDOM_SEED
from river.datasets import synth
from unittest.mock import patch
from semisupervised_methods.clustream_and_label import CluStreamMicroClusterWithLabel, CluserAndLabel

class CluStreamMicroClusterWithLabelTests(unittest.TestCase):

    def setUp(self):
        self.dataset = synth.Hyperplane(seed=RANDOM_SEED, n_features=2)
  

    def test_add_label(self):
        labels = {'A': 2, 'B': 1}
        x,y = next(self.dataset.take(1))
        mc = CluStreamMicroClusterWithLabel(x = x,labels=labels,w = 1)

        mc._add_label('C')
        mc._add_label('B')

        expected_labels = {'A': 2, 'B': 2, 'C': 1}
        self.assertEqual(mc.labels, expected_labels)

    def test_insert(self):
        x,_ = next(self.dataset.take(1))
        y1 = 1
        y2 = 2
        w = 1
        timestamp = 2
        mc = CluStreamMicroClusterWithLabel(x = x, labels = {y1:1})

        mc.insert(x,y2, w, timestamp)
        expected_labels = {y1: 1,y2:1}
        self.assertEqual(mc.labels, expected_labels)
        self.assertEqual(mc.timestamp,timestamp)

        mc.insert(x,y1, w, timestamp+1)
        expected_labels = {y1: 2,y2:1}
        self.assertEqual(mc.labels, expected_labels)
        self.assertEqual(mc.timestamp,timestamp+1)

class CluserAndLabelTests(unittest.TestCase):
    def setUp(self):
        self.dataset = synth.Hyperplane(seed=RANDOM_SEED, n_features=2)
    def test_merge_clusters_label_count(self):
        labels1 = {'A': 2, 'B': 1}
        labels2 = {'B': 3, 'C': 2}
        cluster = CluserAndLabel()

        merged_labels = cluster._merge_clusters_label_count(labels1, labels2)

        expected_labels = {'A': 2, 'B': 4, 'C': 2}
        self.assertEqual(merged_labels, expected_labels)

    def test_learn_one_no_enough_clusters(self):
     
        cluster = CluserAndLabel(max_micro_clusters = 4)
        for x,y in self.dataset.take(3):
            cluster = cluster.learn_one(x,y)
        x,y = next(self.dataset.take(1))

        self.assertFalse(cluster._initialized)
        cluster.learn_one(x, y = y)

        self.assertEqual(len(cluster.micro_clusters), 4)
        self.assertTrue(cluster._initialized)
        self.assertEqual(cluster.sum_labels(), 4)
        self.assertEqual(max([c.timestamp for c in cluster.micro_clusters.values()]),3)

    def test_learn_one_enough_clusters(self):
     
        cluster = CluserAndLabel(max_micro_clusters = 3)
        for x,y in self.dataset.take(3):
            cluster = cluster.learn_one(x,y)
        x,y = next(self.dataset.take(1))

        self.assertTrue(cluster._initialized)
        cluster.learn_one(x, y = y)
        cluster.learn_one(x, y = y)

        self.assertEqual(len(cluster.micro_clusters), 3)
        self.assertTrue(cluster._initialized)
        self.assertEqual(cluster.sum_labels(), 5)
        self.assertEqual(max([c.timestamp for c in cluster.micro_clusters.values()]),4)

    def test_maintain_micro_cluster_with_old(self):
        cluster = CluserAndLabel(max_micro_clusters = 2, time_window=3)
        for i,(x,y) in enumerate(self.dataset.take(2)):
            cluster = cluster.learn_one(x,i)

        cluster._timestamp = 5
        x,y = next(self.dataset.take(1))
        cluster._maintain_micro_clusters(x,w = 1,y = 3)
        self.assertEqual(cluster.sum_labels(), 2)
        self.assertTrue(3 in cluster.micro_clusters[0].labels.keys())


    def test_maintain_micro_cluster_no_old(self):
        cluster = CluserAndLabel(max_micro_clusters = 2, time_window=10)
        for i,(x,y) in enumerate(self.dataset.take(2)):
            cluster = cluster.learn_one(x,i)
        cluster._maintain_micro_clusters(x,w = 1,y = 3)
        self.assertEqual(cluster.sum_labels(), 3)
        self.assertTrue(3 in cluster.micro_clusters[0].labels.keys())
        self.assertTrue(1 in cluster.micro_clusters[1].labels.keys())
        self.assertTrue(0 in cluster.micro_clusters[1].labels.keys())

    def test_predict_one(self):
        cluster = CluserAndLabel(max_micro_clusters = 2, time_window=3)
        for i,(x,y) in enumerate(self.dataset.take(2)):
            cluster = cluster.learn_one(x,i)

        prediction = cluster.predict_one(x)

        self.assertEqual(prediction, 1)

    def test_predict_proba_one(self):
        cluster = CluserAndLabel(max_micro_clusters = 3, time_window=10)
        for x,y in [(2,1),(2,2),(3,1),(4,2),(3,2)]:
            cluster = cluster.learn_one({1:x},y)

        prediction = cluster.predict_proba_one({1:2})

        self.assertEqual(max(prediction.values()), 1/2)
