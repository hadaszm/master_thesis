import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        self.scaler = StandardScaler() #TODO: check how the river scaler is working and consider swapping it somehow
        self.classifier = riverKNN(n_neighbors=1)
        self.required_ratio = required_ratio #TODO: maybe diffrent stopping criteria

    def _label_the_closest(self):
        min_ind = np.argmin(np.array(self.U.apply(lambda x: self.classifier._nn.find_nearest((x,None))[0][1],axis=1 )))
        x = self.U.iloc[min_ind,:]
        self.classifier = self.classifier.learn_one(x,1)
        self.L = pd.concat([self.L,pd.DataFrame(x).T])
        self.U = self.U.drop(index=x.name)
        #U.reset_index(inplace=True,drop = True) TODO: maybe the forgetting window mechanism will be helpful TODO: read more how it is done on river implemenation 
  

    def _train_classifier(self):
        self.scaler.fit(pd.concat([self.L,self.U]))
        self.L = pd.DataFrame(self.scaler.transform(self.L))
        self.U = pd.DataFrame(self.scaler.transform(self.U))
        for _, x in self.L.iterrows(): # TODO: think how to avoid loop
            self.classifier = self.classifier.learn_one(x,1)
            
        while (len(self.L)/(len(self.L)+len(self.U)))<self.required_ratio:
            self._label_the_closest()

    def learn_one(self,x,y):
        if self._timestamp < self.training_time: #TODO: change to learn in every iterration
            if y == self.positive:
                self.L = pd.concat([self.L, pd.DataFrame(x, index = [self._timestamp])])
            else:
                self.U = pd.concat([self.U, pd.DataFrame(x, index = [self._timestamp])])

        elif self._timestamp == self.training_time:  
            self._train_classifier()

        else:
            cols = list(x.keys())
            x = self.scaler.transform(pd.DataFrame(x,index=[0]))
            x = dict(zip(cols,x[0]))
            self.classifier = self.classifier.learn_one(x,int(y==self.positive))
        self._timestamp+=1
        return self

    def predict_one(self,x):
        cols = list(x.keys())
        x = self.scaler.transform(pd.DataFrame(x,index=[0]))
        x = dict(zip(cols,x[0]))
        dist =  self.classifier._nn.find_nearest((x,None))[0][1]
        if dist < self.threshold:
            return self.positive
        return self.negative

