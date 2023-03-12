from collections import Counter, defaultdict
import math
import typing
from river.cluster.clustream import CluStream
from river.cluster.clustream import CluStreamMicroCluster


class CluStreamMicroClusterWithLabel(CluStreamMicroCluster):
    """ 
    Extension of CluStreamMicroCluster class


    Parameters
    ----------
        labels: dict
            dict with number of each label occurnace

        the rest of parameters as in CluStreamMicroCluster class

    """
    def __init__(   self,   x: dict = defaultdict(float),
        labels: dict = {},
        w: float = None,
        timestamp: int = None ):
            super().__init__(x,w,timestamp)
            self.labels = labels
                
    def _add_label(self,y):
        """ add the label y occurance"""
        self.labels[y] = self.labels.get(y,0)+1

    def insert(self, x,y, w, timestamp):
        """insert new instance to a microcluster"""
        self.var_time.update(timestamp, w)
        for x_idx, x_val in x.items():
            self.var_x[x_idx].update(x_val, w)
        self._add_label(y)




class CluserAndLabel(CluStream):
    """ 
    Extension of CluStream class
        - instead of CluStreamMicroCluster a CluStreamMicroClusterWithLabel used
        - modified  _maintain_micro_clusters, predict_one, learn_one
        - macroclusters not used

    Parameters
       ----------
        train_period: int
            number of labelled instances the classifier needs to get before it can work with unlabelled ones

        the rest of parameters as in CluStream class

       """

    def __init__(self,train_period=0,
                 n_macro_clusters: int = 5,
                 max_micro_clusters: int = 100,
                 micro_cluster_r_factor: int = 2,
                 time_window: int = 1000,
                 time_gap: int = 100,
                 seed: int = None,
                 **kwargs, ):
        super().__init__(n_macro_clusters, max_micro_clusters,
                         micro_cluster_r_factor, time_window, time_gap, seed, **kwargs)
        self.micro_clusters: typing.Dict[int,
                                         CluStreamMicroClusterWithLabel] = {}
        self.train_period=train_period


    def _merge_clusters_label_count(self, labels1, labels2):
        """
        When two clusters are merged their labels dictionary also need to be merged
        and if they have the same keys, the value need to be summed

        Parameters
        ----------
        labels1,labels2: dict
            dictionaries of labels as keys and their frequency as values

        Returns
        -------
        dict
            with sum of each label frequency
        """

        cnt1 = Counter(labels1)
        cnt2 = Counter(labels2)
        return dict(cnt1 + cnt2)

    def _maintain_micro_clusters(self, x, w, y):
        """
        Diffrence in merging introduced
        TODO: better understandf what w does
        """
        # Calculate the threshold to delete old micro-clusters
        threshold = self._timestamp - self.time_window

        # Delete old micro-cluster if its relevance stamp is smaller than the threshold
        del_id = None
        for i, mc in self.micro_clusters.items():
            if mc.relevance_stamp(self.max_micro_clusters) < threshold:
                del_id = i
                break

        if del_id is not None:
            self.micro_clusters[del_id] = CluStreamMicroClusterWithLabel(
                x=x,
                w=w,
                labels={y: 1},
                timestamp=self._timestamp,
            )
            return

        # Merge the two closest micro-clusters
        closest_a = 0
        closest_b = 0
        min_distance = math.inf
        for i, mc_a in self.micro_clusters.items():
            for j, mc_b in self.micro_clusters.items():
                if i <= j:
                    continue
                dist = self._distance(mc_a.center, mc_b.center)
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j

        # diffrent merging - also labels count needs to be added
        labels_merged = self._merge_clusters_label_count(self.micro_clusters[closest_a].labels,
                                                         self.micro_clusters[closest_b].labels)
        self.micro_clusters[closest_a] += self.micro_clusters[closest_b]
        self.micro_clusters[closest_a].labels = labels_merged
        self.micro_clusters[closest_b] = CluStreamMicroClusterWithLabel(
            x=x,
            w=w,
            labels={y: 1},
            timestamp=self._timestamp,
        )

    # def return_microclusters(self):
    #     """ Method for printing the labels statistics in each microcluster"""
    #     for i, mc in self.micro_clusters.items():
    #         print(i,mc.labels)
    def sum_labels(self):
        """ Method for summing the number of stored labels -- needed for testing"""
        s = 0
        for i, mc in self.micro_clusters.items():
            s+=sum([v for v in mc.labels.values()])
        return s

    def predict_one(self, x):
        """ Prediction yields the majority class in a microcluser x belongs to """
        cluster_num = self._get_closest_mc(x)[0]
        labels = self.micro_clusters[cluster_num].labels
        return max(labels, key=labels.get)

    def learn_one(self, x, y=None, w=1):
        """ Learns y as well (orginal algorithm was an unsupervised one).
        If y not avaiable it predicts a label and assigns it as a pseudolabel
        The macroclusters are not used"""
        if y is None:
            if self._timestamp < self.train_period:
                return self
            y = self.predict_one(x)
        
        self._timestamp += 1

        if not self._initialized:
            self.micro_clusters[len(self.micro_clusters)] = CluStreamMicroClusterWithLabel(
                x=x,
                w=w,
                labels={y: 1},
                # When initialized, all micro clusters generated previously will have the timestamp reset to the current
                # time stamp at the time of initialization (i.e. self.max_micro_cluster - 1). Thus, the timestamp is set
                # as follows.
                timestamp=self.max_micro_clusters - 1,
            )

            if len(self.micro_clusters) == self.max_micro_clusters:
                self._initialized = True

            return self

            # Determine the closest micro-cluster with respect to the new point instance
        closest_id, closest_dist = self._get_closest_mc(x)
        closest_mc = self.micro_clusters[closest_id]

        # Check whether the new instance fits into the closest micro-cluster
        if closest_mc.weight == 1:
            radius = math.inf
            center = closest_mc.center
            for mc_id, mc in self.micro_clusters.items():
                if mc_id == closest_id:
                    continue
                distance = self._distance(mc.center, center)
                radius = min(distance, radius)
        else:
            radius = closest_mc.radius(self.micro_cluster_r_factor)

        if closest_dist < radius:
            closest_mc.insert(x, w, y, self._timestamp)
            return self

        # If the new point does not fit in the micro-cluster, micro-clusters
        # whose relevance stamps are less than the threshold are deleted.
        # Otherwise, closest micro-clusters are merged with each other.
        self._maintain_micro_clusters(x=x, w=w, y=y)
        return self
