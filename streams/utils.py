from functools import partial
from operator import is_not
import numpy as np

def FL(stream) -> bool:
    """
    #TODO: maybe change the implementaion
    Indicate if the instance is labelled

    Parameters
    ----------
    stream: list of tuple
        list of instances x, and corresponing labels y (possible that it is None) #TODO: check if accurate assumption
        The x must be a first element of a tuple

    Returns
    -------
    list of tuples
        list of labelled tuples

    """
    def hasLabel(instance):
        '''
        If the instance does not have label return None, otherwise it returns an instance
        '''
        if instance[1] is None:
            return None
        return instance

    return list(filter(partial(is_not, None), list(map(partial(hasLabel), stream))))


def FU(stream, probability) -> list:
    """
    Remove the label with the given probability

    Parameters
    ----------
    stream: list of tuple
        list of instances x, and corresponing labels y (possible that it is None) #TODO: check if accurate assumption
        The x must be a first element of a tuple
    probability: float
        probability that the label is removed

    Returns
    -------
    list tuple
        list of  labelled or unlabelled instnaces

    """
    def chceck_proba(instance):
        '''deletes label if random value is smaller then the probability'''
        if np.random.random() < probability:
            return instance[0], None
        return instance

    stream = list(map(partial(chceck_proba), stream))
    return stream

