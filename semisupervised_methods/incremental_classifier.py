import operator
from river.base.classifier import Classifier as RiverClassifer


class IncrementalClassifer(RiverClassifer):
    """ Add possibility to predict on unlabel instances"""

    def __init__(self, threshold, classifier, params):
         """
         TODO: implement more if needed or add some kind of decorator
        Parameters
        ----------
        threshold : float
            threshold from which the prediction is confident enough to
            use it for training

        classifer: RiverClassifer (probably)
            base classifer, needs to have a predict_proba (TODO: chceck what if not)
        params: dict
            parameters for the classifier

        """
         self.threshold = threshold
         self.classifier = classifier(**params)

         super().__init__() 

    def learn_one(self,x,y=None):
        """ 
         add possibility of learning without label
         if prediction is not confident enough the instance is ommited
        """
        if y is None:
            pred_class,pred_proba =  max(self.classifier.predict_proba_one(x).items(), key=operator.itemgetter(1))
            if pred_proba >= self.threshold: 
                y = pred_class
            else:
                return self
            
        self.classifier = self.classifier.learn_one(x=x,y=y)
        return self
    
    def predict_one(self, x):
        return self.classifier.predict_one(x=x)
    
    def predict_proba_one(self, x):
        return self.classifier.predict_proba_one(x=x)
