import operator
import typing
import inspect
from river.base.classifier import Classifier as RiverClassifer
from river.base.base import Base


class IncrementalClassifer(RiverClassifer):
    """ Add possibility to predict on unlabel instances"""

    def __init__(self, threshold, classifier, train_period=0, params={}):
        """
       Parameters
       ----------
        threshold : float
           threshold from which the prediction is confident enough to
           use it for training

        classifer: RiverClassifer 
           base classifer, needs to have a predict_proba 

        train_period: int
            number of labelled instances the classifier needs to get before it can work with unlabelled ones

        params: dict
           parameters for the classifier

       """
        super().__init__()
        self.threshold = threshold
        self.classifier = classifier(**params)
        self._timestamp = 0
        self.iter_number = 0
        self.train_period = train_period

        super().__init__()

    def _check_max_probabaility(self, x):
        '''
        return label with the highest proability
        '''
        return max(self.classifier.predict_proba_one(
            x).items(), key=operator.itemgetter(1))

    def learn_one(self, x, y=None):
        """ 
        add possibility of learning without label
        if prediction is not confident enough the instance is ommited
        
        Parameters
       ----------
        x : dict
            instance to be learned

        y: int (optional)
           label 

        """
        if y is None:
            if self._timestamp < self.train_period:
                return self

            pred_class, pred_proba = self. _check_max_probabaility(x)
            if pred_proba >= self.threshold:
                y = pred_class
            else:
                return self

        self.classifier = self.classifier.learn_one(x=x, y=y)
        self._timestamp += 1
        return self

    def predict_one(self, x):
        return self.classifier.predict_one(x=x)

    def predict_proba_one(self, x):
        return self.classifier.predict_proba_one(x=x)

    def _get_params(self) -> typing.Dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""

        params = {}

        # type: ignore
        for name, param in inspect.signature(self.classifier.__init__).parameters.items():

            # Keywords parameters
            attr = getattr(self.classifier, name)
            params[name] = attr

        params['threshold'] = self.threshold
        params['train_period'] = self.train_period
        params['classifier'] = self.classifier.__class__

        return params
