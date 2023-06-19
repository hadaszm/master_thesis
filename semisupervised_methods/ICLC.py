import datetime
import typing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from river.base.classifier import Classifier as RiverClassifer
from numpy import histogram
import pandas as pd
import inspect


class ICLC(RiverClassifer):
    def __init__(self, classifer, clustering_method, classifier_params, clustering_params, nu, drift_detector):
        """

        Parameters
        ----------

        classifer: RiverClassifer (probably)
            base classifer

        clustering_method: Sklearn style clustering method
            clustering method for unlabelled instances

        classifier_params: dict
            parameters for the classifier

        clustering_params: dict
            parameters for the clustering method

        drift_detector: River Drift Detector
            drift detector, if drift detected the clustering_method is reintialized

        nu: int
            after how many unlabelled instances the prediction on centers are made 
        """
        super().__init__()
        self.counter = 0  # counter of the labelled instances
        self.classifier = classifer(**classifier_params)
        self.clustering_method_type = clustering_method
        self.clustering_method = self.clustering_method_type(**clustering_params)        
        self.clustering_params = clustering_params
        self.nu = nu
        self.drift_detectors = []
        self.drift_detector = drift_detector
        self._timestamp = 0
        self._unlabelled_instances_cnt = 0
        self.unlabelled_instances = []


    def _learn_from_unlabelled(self, columns):
        """
        Train the classifer on unlabelled instances by 
        1. predicting the psudolabels
        2. fed the classifier with that intances + pseudolabels

        Parameters
        ----------
        columns: list 
            names of columns in the data stream
        """

        predicted_labels = []
        for center in self.clustering_method.center:
            x = dict(zip(columns, center))
            y = self.classifier.predict_one(x)
            predicted_labels.append((x, y))

        for x, y in predicted_labels:
            self._timestamp += 1
            self.classifier = self.classifier.learn_one(x, y)

    def _init_drift_detectors(self,x):
        """
       For each column seperate drift detector is initialized

        Parameters
        ----------
        x: dict 
            instance 

        """
        self.drift_detectors = [self.drift_detector() for _ in range(len(x.keys()))]

    def _check_if_drift_detetcted(self):
        '''
        checks if any drift detector detected drift
        '''
        return any([dd.drift_detected for dd in self.drift_detectors ])


    def _update_drift_detector(self,dd,x):
        dd.update(x)

    def learn_one(self, x, y=None):
        """
       Function for learning a new instance by classifier. If drift is detected the clustering method is reinitialized.
       If y is not None the classifier is taught with it, otherwise the clustering method is fed with the x.

        Parameters
        ----------
        x: dict 
            intsance 
        y: int, optional
            label of the instance
        """
        if not self.drift_detectors:
            self._init_drift_detectors(x)
        
        map(self._update_drift_detector,self.drift_detectors,
            [list(x.values())[i] for i in range(len(self.drift_detectors))])
        

        if self._check_if_drift_detetcted():
            self.clustering_method = self.clustering_method_type(**self.clustering_params)
            self._unlabelled_instances_cnt = 0
            return self
        if y is None:
            self._unlabelled_instances_cnt += 1
            self.clustering_method = self.clustering_method.learn_one(x)
            return self
        else:
            self._timestamp += 1

            self.classifier = self.classifier.learn_one(x=x, y=y)
            if self._unlabelled_instances_cnt>0 and self._unlabelled_instances_cnt % self.nu == 0:
                self._learn_from_unlabelled(list(x.keys()))
            return self

    def predict_one(self, x):
        return self.classifier.predict_one(x=x)

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """
        Returns the parameters that were used during initialization
        """

        params = {}

        for name, param in inspect.signature(self.classifier.__init__).parameters.items():
            # Keywords parameters
            attr = getattr(self.classifier, name)
            params[f"classifier_" + str(name)] = attr

        
        for name, param in inspect.signature(self.clustering_method.__init__).parameters.items():
            # Keywords parameters
            attr = getattr(self.clustering_method, name)
            params[f"clustering_" + str(name)] = attr

        params['clustering'] = self.clustering_method.__class__
        params['classifier'] = self.classifier.__class__
        params['nu'] = self.nu

        return params
