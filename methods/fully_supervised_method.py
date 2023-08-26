import operator
import typing
import inspect
from river.base.classifier import Classifier as RiverClassifer
from river.base.base import Base


class FullySupervisedClassifer(RiverClassifer):
    """ Add possibility of using the fully supervised RiverClassifer"""

    def __init__(self, classifier, train_period=0, params={}):
        """
       Parameters
       ----------

        classifer: RiverClassifer 
           base classifer, needs to have a predict_proba 

        train_period: int
            number of labelled instances the classifier needs to get before it can work with unlabelled ones

        params: dict
           parameters for the classifier

       """
        super().__init__()
        self.classifier = classifier(**params)
        self._timestamp = 0
        self.iter_number = 0
        self.train_period = train_period

        super().__init__()


    def learn_one(self, x, y=None):
        """ 
        pass this method to the classifer instance
        
        Parameters
       ----------
        x : dict
            instance to be learned

        y: int 
           label 

        """
        if y is None:
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

        params['train_period'] = self.train_period
        params['classifier'] = self.classifier.__class__

        return params
