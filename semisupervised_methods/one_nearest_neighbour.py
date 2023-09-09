import pandas as pd
import numpy as np
from river.preprocessing import StandardScaler
from river.neighbors import KNNClassifier as riverKNN
import inspect
import typing

class oneNNClassifer():
    def __init__(self, training_time = 1000, threshold = 1,positive =1,negative = 0,required_ratio = 0.75) -> None:
        
        self.threshold = threshold
        self._timestamp = 0
        self.training_time =training_time
        self.L = pd.DataFrame() # labelled instances
        self.U = pd.DataFrame() # unlabelled instances
        self.positive = positive # which class is considered to be positive 
        self.negative = negative
        self.scaler = StandardScaler() 
        self.classifier = riverKNN(n_neighbors=1)
        self.required_ratio = required_ratio 

    def _label_the_closest(self):
        """
        Find the nearest labelled neighbour to every unlabelled instance, label the one whose neighbour is the closest.
        Then teach the classifer on it.

        """
        # 
        # find the index of the instance, that will be labelled
        min_ind = np.argmin(np.array(self.U.apply(lambda x: self.classifier._nn.find_nearest((x,None))[0][1],axis=1 )))
        x = self.U.iloc[min_ind,:]
        # train the classifer
        self.classifier = self.classifier.learn_one(x,1)
        # add the newly labelled instance to the set of labelled instances
        self.L = pd.concat([self.L,pd.DataFrame(x).T])
        # delete the newly labelled instance from the set of unlabelled instances
        self.U = self.U.drop(index=x.name)
        
  

    def _train_classifier(self):
        """
        Train the classifer with all labelled instances.
        Until achieving the desire ratio of labelled instances, label the closest unlabelled one.
        """

        self.L = pd.DataFrame(self.scaler.transform_many(self.L))
        self.U = pd.DataFrame(self.scaler.transform_many(self.U))


        for _, x in self.L.iterrows(): 
            self.classifier = self.classifier.learn_one(x,1)
            
        # until desired ratio is acheived label new instances 
        while (len(self.L)/(len(self.L)+len(self.U)))<self.required_ratio:
            self._label_the_closest()
        self.L = pd.DataFrame()
        self.U = pd.DataFrame()

    def learn_one(self,x,y=None):
        """ 
        If timestanp is smaller, then the training time add instances to the appropriate set.
        Otherwise, if y avaiable teach the classifer with it.
        
        Parameters
       ----------
        x : dict
            instance to be learned

        y: int (optional)
           label 

        """
        self.scaler.learn_one(x)
        if self._timestamp <= self.training_time: 
            if y == self.positive:
                self.L = pd.concat([self.L, pd.DataFrame(x, index = [self._timestamp])])
                self._timestamp+=1
            else:
                self.U = pd.concat([self.U, pd.DataFrame(x, index = [self._timestamp])])

            if self._timestamp == self.training_time:  
                self._train_classifier()

        elif y==self.positive:
            x = self.scaler.transform_one(x)
            self.classifier = self.classifier.learn_one(x,self.positive)
            self._timestamp+=1
        return self

    def predict_one(self,x):
        """ 
        If the instance has a labelled neighbour close enough predict positive class, otherwise negative.
        
        Parameters
       ----------
        x : dict
            instance to be learned

        Returns
        ----------
        prediction

        """
        x = self.scaler.transform_one(x)
        dist =  self.classifier._nn.find_nearest((x,None))
        if len(dist) >0 and dist[0][1] < self.threshold:
            return self.positive
        return self.negative
    

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """
        Returns the parameters that were used during initialization
        """

        params = {}
        params['threshold'] = self.threshold
        params['training_time']=self.training_time
        params['positive']=self.positive 
        params['negative']=self.negative
        params['required_ratio'] = self.required_ratio 

        return params

