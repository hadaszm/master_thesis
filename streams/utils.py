from functools import partial
from operator import is_not
import numpy as np
from streams.stream_section import StreamSection
from river.preprocessing import StandardScaler


def generate_stream_section(dataset, stream_name, dataset_name, with_scaling = False, start=0, stop=1000):
    """
    generates the StreamSection 
    if synth geneartor for take the stop-start instances are taken
    if needed scaling added

    """
    #TODO: add exception
    # TODO: Maybe move to some globalk constants
    if dataset_name in ['LED', 'AGRAWL', 'RandomRBF','HyperPlane']:
        to_take = stop-start
        if not with_scaling:
            return StreamSection(stream_name, [(cur_idx, cur_idx, instance[0], instance[1]) for cur_idx, instance in enumerate(dataset.take(to_take))], True)
        else:
            scaler = StandardScaler()
            res = []
            for cur_index,(x,y) in enumerate(dataset.take(to_take)):
                scaler = scaler.learn_one(x)
                x = scaler.transform_one(x)
                res.append((cur_index,cur_index,x,y))
            return StreamSection(stream_name, res, True)
    elif dataset_name in ['Airlines', 'Cover_Type', 'Electricity']:
        # dataset pass as list
        if not with_scaling:
            return StreamSection(stream_name, [(cur_idx, cur_idx, instance[0], instance[1]) for cur_idx, instance in enumerate(dataset[start:stop])], True)
        else:
            scaler = StandardScaler()
            res = []
            for cur_index,(x,y) in  enumerate(dataset[start:stop]):
                scaler = scaler.learn_one(x)
                x = scaler.transform_one(x)
                res.append((cur_index,cur_index,x,y))
            return StreamSection(stream_name, res, True)
    else:
        to_take = stop-start
        return StreamSection(stream_name, [(cur_idx, cur_idx, instance[0], instance[1]) for cur_idx, instance in enumerate(dataset.take(to_take))], True)



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
        if instance[3] is None:
            return None
        return instance

    return list(filter(partial(is_not, None), list(map(partial(hasLabel), stream))))


def FU(stream, probability, no_delete_period=10) -> list:
    """
    Remove the label with the given probability

    Parameters
    ----------
    stream: list of tuple
        list of instances x, and corresponing labels y (possible that it is None) #TODO: check if accurate assumption
        The x must be a first element of a tuple
    probability: float
        probability that the label is removed
    no_delete_period: int
        times, when the labels are not removed

    Returns                                         
    -------
    list tuple
        list of  labelled or unlabelled instnaces

    """
    def chceck_proba(instance):
        '''deletes label if random value is smaller then the probability'''
        if instance[0] > no_delete_period and np.random.random() < probability:
            return instance[0], instance[1], instance[2], None
        return instance

    stream = list(map(partial(chceck_proba), stream))
    return stream
