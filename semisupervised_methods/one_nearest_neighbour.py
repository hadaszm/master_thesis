import pandas as pd
import numpy as np
from river.preprocessing import StandardScaler
from river.neighbors import KNNClassifier as riverKNN

class oneNNClassifer():
    def __init__(self, training_time = 1000, threshold = 1,positive =1,negative = 0,required_ratio = 0.75) -> None:
        self.threshold = threshold
        self._timestamp = 0
        self.training_time =training_time
        self.L = pd.DataFrame()
        self.U = pd.DataFrame()
        self.positive = positive # which class is considered to be positive 
        self.negative = negative
        self.scaler = StandardScaler() 
        self.classifier = riverKNN(n_neighbors=1)
        self.required_ratio = required_ratio 

    def _label_the_closest(self):
        min_ind = np.argmin(np.array(self.U.apply(lambda x: self.classifier._nn.find_nearest((x,None))[0][1],axis=1 )))
        x = self.U.iloc[min_ind,:]
        self.classifier = self.classifier.learn_one(x,1)
        self.L = pd.concat([self.L,pd.DataFrame(x).T])
        self.U = self.U.drop(index=x.name)
        
  

    def _train_classifier(self):
        self.L = pd.DataFrame(self.scaler.transform_many(self.L))
        self.U = pd.DataFrame(self.scaler.transform_many(self.U))


        for _, x in self.L.iterrows(): 
            self.classifier = self.classifier.learn_one(x,1)
            
        while (len(self.L)/(len(self.L)+len(self.U)))<self.required_ratio:
            self._label_the_closest()
        del self.L, self.U

    def learn_one(self,x,y=None):
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
        x = self.scaler.transform_one(x)
        dist =  self.classifier._nn.find_nearest((x,None))
        if len(dist) >0 and dist[0][1] < self.threshold:
            return self.positive
        return self.negative

