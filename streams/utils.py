from functools import partial
from operator import is_not
import numpy as np
from streams.stream_section import StreamSection
from river.preprocessing import StandardScaler


def generate_stream_section(datastream, stream_name, dataset_type = 'real', with_scaling = False, start=0, stop=1000):
    """
    The function generates the StreamSection. It enables three types of streams:
        - synth - streams from river, the instances are obtained by take function
        - real - the stream is passed as list of tuples instance, label
        - delayed the stream is passed as a list of tuples (init index, current index, instance, label)
    Scaling instances by standard scaller is possible.
    If the dataset_type is synth the number of instances, that are taken is stop - start
    Parameters
    ----------
    datasetream: list of tuples / river.synth datasetream
        datasetream to be processed
    stream_name: str
        name of the datastream
    dataset_type: str
        type of the stream. Three possibilities delayed,real and synth
    with_scaling: bool
        Should the scaling with StandardScaler be performed 
    start: int
        from which place in the list should the stream start (applicable when dataset_type is real)
    stop = int
        up to which place should the instances be taken (applicable when dataset_type is real)
        

    Returns
    -------
    list of tuples
        list of  init_index, cur_idx, instance, labvel
    """
    if dataset_type == 'synth':
        # determine how many instances to take
        to_take = stop-start

        # scale if requested
        if not with_scaling:
            return StreamSection(stream_name, [(cur_idx, cur_idx, x,y) for cur_idx, (x,y) in enumerate(datastream.take(to_take))], True)
        else:
            scaler = StandardScaler()
            res = []
            for cur_index,(x,y) in enumerate(datastream.take(to_take)):
                scaler = scaler.learn_one(x)
                x = scaler.transform_one(x)
                res.append((cur_index,cur_index,x,y))
            return StreamSection(stream_name, res, True)
    elif dataset_type == 'real':
         # scale if requested
        if not with_scaling:
            return StreamSection(stream_name, [(cur_idx, cur_idx, x,y) for cur_idx, (x,y) in enumerate(datastream[start:stop])], True)
        else:
            scaler = StandardScaler()
            res = []
            for cur_index,(x,y) in  enumerate(datastream[start:stop]):
                scaler = scaler.learn_one(x)
                x = scaler.transform_one(x)
                res.append((cur_index,cur_index,x,y))
            return StreamSection(stream_name, res, True)
    elif dataset_type == 'delayed':
        if not with_scaling:
            return StreamSection(stream_name, [( init_ind, cur_ind,x,y) for  init_ind, cur_ind,x,y in datastream[start:stop]], True)
        else:
            scaler = StandardScaler()
            res = []
            for  init_ind, cur_ind,x,y in datastream[start:stop]:
                scaler = scaler.learn_one(x)
                x = scaler.transform_one(x)
                res.append((init_ind, cur_ind,x,y))
            return StreamSection(stream_name, res, True)
    else:
        raise ValueError('dataset_type should be real or synth')
        



def FL(stream) -> bool:
    """
    
    Indicate if the instance is labelled

    Parameters
    ----------
    stream: list of tuple
        list of instances x, and corresponing labels y (possible that it is None)
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
        list of instances x, and corresponing labels y (possible that it is None) 
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
