import operator
from streams.utils import FL, FU, generate_stream_section
from streams.stream_section import StreamSection
from river import metrics
from multiprocessing.pool import ThreadPool
from constants import NUMBER_OF_THREADS, FREQUENCY_OF_PREDICTIONS
import numpy as np
import logging
import datetime
from river.utils import Rolling


def get_most_frequent(predictions):
    ''' 
    Get the most common predction

    Parameters
    ----------
    predictions: list
        The list of predicted labels

    Returns
    ---------
    int
        The most popular class

    '''
    return max(set(predictions), key=predictions.count)


def update_performance_measures(predictions, true_label, B, metrics):
    ''' 
    The prediction for each awaiting (for a label) instance is made every K iterations 

    Parameters
    ----------
    predictions: dict
        The dictionary of predctions 

    true_label: int
        The true class of an instance

    B: int
        Number of bins 

    metrics: dict
        The dictionary holding the metrics to calculate the results
    '''
    interval = (len(predictions)-1)/B
    preds = list(predictions.values())
    previously_pred = None # the prediction of the previous bin
    for b in range(B+2):
        if b == 0:
            metrics[b].update(true_label, preds[0])
            previously_pred = preds[0]
        elif b == B+1:
            metrics[b].update(true_label, preds[-1])
        else:
            start = int(1+interval*(b-1))
            stop = int(1+interval*(b))
            if stop == len(preds):
                stop -= 1
            interval_preds = preds[start:stop]
            if interval_preds:
                y_pred = get_most_frequent(interval_preds)
                previously_pred = y_pred
                metrics[b].update(true_label, y_pred)
            else:
                metrics[b].update(true_label, previously_pred)


def make_prediction_for_awaiting(h, cur_idx, P, L, K):
    ''' 
    The prediction for each awaiting (for a label) instance is made every K iterations 

    Parameters
    ----------
    h: classifier
        The trained classifer that predicts the class

    cur_idx: int
        The index of a currently processed observation

    P: dict
        The dictionary with predictions 

    L: dict
        The dictionary with awaiting results

    K: int
        The number indicates how often the predction is made

    '''
    # foreach instance in L add predictionin time cur_idx in P
    for idx, instance in L.items():
        # make prediction every K instances
        if abs(idx-cur_idx) % K == 0:
            P[idx][cur_idx] = h.predict_one(instance)


def add_delay_constant(stream, delay, no_delete_period, dataset_name, q):
    ''' 
    Add the constant delay for each instance from the stream

    Parameters
    ----------
    stream: list
        List of tuples (int,int,dict,int)

    delay: int
        The length of the delay

    no_delete_period: int
        The amount of the first instances, which should not be delayed

    dataset_name: str
        The name of the dataset

    q: int, int
        The boundries of the stream

    '''
    new_stream = []
    i = 0
    for idx_1, idx_2, x, y, in stream:
        if i < no_delete_period:
            new_stream.append((i, i, x, y))
            i = i+1
            continue
        if (i-no_delete_period) % delay == 0 and i-no_delete_period != 0:
            i += delay
        new_stream.append((i, i, x, None))
        new_stream.append((i+delay, i, x, y))
        i += 1
    new_stream.sort(key=lambda x: x[0])
    return StreamSection(f'{dataset_name}', new_stream, False)


def add_delay_random(stream, max_delay, no_delete_period, dataset_name, q):
    ''' 
    Add the random delay for each instance from the stream. The delay is drawn from (1,max_delay) interval.
    If drawn index is not available the delay is incremented

    Parameters
    ----------
    stream: list
        List of tuples (int,int,dict,int)

    max_delay: int
        The max length of the delay

    no_delete_period: int
        The amount of the first instances, which should not be delayed

    dataset_name: str
        The name of the dataset

    q: int, int
        The boundries of the stream
    '''
    new_stream = []
    used_indexes = []
    i = 0
    for idx_1, idx_2, x, y, in stream:
        if i < no_delete_period:
            used_indexes.append(i)
            new_stream.append((i, i, x, y))
            i = i+1
            continue
        while i in used_indexes:
            i += 1
        delay = np.random.randint(1, max_delay)
        while i+delay in used_indexes:
            delay += 1  # if sampling again infinite loop possible
        new_stream.append((i, i, x, None))
        new_stream.append((i+delay, i, x, y))
        used_indexes.extend([i, i+delay])
        i += 1
    new_stream.sort(key=lambda x: x[0])
    return StreamSection(f'{dataset_name}', new_stream, False)


def add_delay_infinite(stream, delay, no_delete_period, dataset_name, q):
    ''' 
    Add the infinite delay with probaiality delay

    Parameters
    ----------
    stream: list
        List of tuples (int,int,dict,int)

    delay: float
        The proabaility of delay

    no_delete_period: int
        The amount of the first instances, which should not be delayed

    dataset_name: str
        The name of the dataset

    q: int, int
        The boundries of the stream

    '''

    new_stream = []
    i = 0
    for idx_1, idx_2, x, y, in stream:
        r = np.random.random()
        if i < no_delete_period or r > delay:
            new_stream.append((i, i, x, y))
        else:
            new_stream.append((i, i, x, None))
        i += 1
    new_stream.sort(key=lambda x: x[0])
    return StreamSection(f'{dataset_name}', new_stream, False)


def generate_streams(initial_stream, dataset_name, q, probas, delay_type, delay, warm_up_period,make_intial_lfs,logger):
    '''
    generates stream by adding the chosen delay, and by removing labels with the given delay

    Parameters
    ----------
    initial_stream: list
        List of tuples (int,int,dict,int)

    dataset_name: str
        The name of the dataset

    q: int, int
        The boundries of the stream
    
    probas: list
        List of probabilites with which the labels are removed

    delay_type: float
        The proabaility of delay

    warm_up_period: int
        The amount of the first instances, which should not be delayed
    
    make_intial_lfs: bool
        Indictates if from the initial stream the unlabelled instances should be removed
    logger:
        logger

    Returns
    --------
    stream_set: list
        List of newly generated streams

    '''

    stream_set = []  

    stream_set.append(initial_stream)
    if make_intial_lfs:
        lfs_stream = StreamSection(
            f'{dataset_name}_lfs_init_{q[0]}_{q[1]}', FL(initial_stream.stream), True)
        stream_set.append(lfs_stream)
    if delay_type == 1:
        dataset_name += '_constant_delay'
        initial_stream = add_delay_constant(
            initial_stream.stream, delay, warm_up_period, dataset_name, q)        
        stream_set.append(initial_stream)
    elif delay_type == 2:
        dataset_name += '_random_delay'
        initial_stream = add_delay_random(
            initial_stream.stream, delay, warm_up_period, dataset_name, q)
        stream_set.append(initial_stream)
    elif delay_type < 1 and delay_type > 0:
        dataset_name += f'_infinite_delay_{delay_type}'
        initial_stream = add_delay_infinite(
            initial_stream.stream, delay_type, warm_up_period, dataset_name, q)        
        stream_set.append(initial_stream)
    elif delay_type == 0:
        # no added delay so no additional stream is created
        pass
    else:
        logger.warning('No such type of delay')


    for p in probas:
        ssl_stream = StreamSection(f'{dataset_name}_ssl_{p}_{q[0]}_{q[1]}', FU(
            initial_stream.stream, p, warm_up_period), False)
        lfs_stream = StreamSection(
            f'{dataset_name}_lfs_{p}_{q[0]}_{q[1]}', FL(ssl_stream.stream), True)
        stream_set.append(ssl_stream)
        stream_set.append(lfs_stream)
    return stream_set


def train_for_stream(my_stream, methods, methods_params, methods_names,
                     metric_funs, K, B, warm_up_period, max_length,logger):
    
    '''
    Holds training and analysis for a given stream

    Parameters
    ----------
    my_stream: list
        List of tuples (int,int,dict,int)

    methods: list
        List of classification methods

    methods_params: list
        List of dictionaries with methods parameters

    methods_names: list
        List of names of the methods

    metric_funs: list
        List of metrics

    K: int
        frequency of predictions

    B: int
        number of bins

    warm_up_period: int
        The amount of the first instances, which should not be delayed
    
    max_length: int
        max number of predictions through time
    logger:
        logger

    Returns
    --------
    name of the stream
    
    results:
        results obtained for a stream

    '''
    results = {}
    # how many labelled instances appered -> needed for prediction of awaiting examples
    labelled_insances_cnt = 0
    logger.debug(f" Start processing {my_stream.__name__}")
    
    for mi, method in enumerate(methods):
        # initilaze method and variables
        results[methods_names[mi]] = {}
        try:
            m = method(**methods_params[mi])
        except Exception as e:
            logger.warning('Cannot initialize the method')
            continue
        for metric_fun in metric_funs:
            metrics = [metric_fun() for _ in range(B+2)]
            periodc_metric = Rolling(
                metric_fun(), window_size=FREQUENCY_OF_PREDICTIONS)

            final_pred_history = []
            h = m
            L = {}
            P = {}
            # preds = []
            logger.debug(f" Start processing method {methods_names[mi]}")
            try:
                for cur_idx, init_idx, x, y in my_stream.stream:
                    # unlabelled instance
                    if y is None:
                        # add instnace and index
                        L[cur_idx] = x
                        P[cur_idx] = {}
                        # predict if after warm up period
                        if m._timestamp > warm_up_period:
                            P[cur_idx][cur_idx] = h.predict_one(x)
                            h = h.learn_one(x)

                    # labelled instance
                    else:
                        labelled_insances_cnt += 1
                        if m._timestamp > warm_up_period:
                            periodc_metric.update(y, h.predict_one(x))
                            if cur_idx != init_idx and init_idx in P.keys():  # delayed label
                                P[init_idx][cur_idx] = h.predict_one(x)
                                L.pop(init_idx)

                                update_performance_measures(
                                    P[init_idx], y, B, metrics)
                            else:  
                                if m._timestamp > warm_up_period:  # if in warmup period the prediction cannot be made
                                    # if it was not delayed only last prediction exists
                                    metrics[B+1].update(y, h.predict_one(x))

                            make_prediction_for_awaiting(h, labelled_insances_cnt, P, L, K)
                        h = h.learn_one(x, y)
                        if labelled_insances_cnt % FREQUENCY_OF_PREDICTIONS == 0 and labelled_insances_cnt > 0:
                            final_pred_history.append(periodc_metric.get())
            except Exception as e:
                logger.warning(f"For metric:{metric_fun} and method: {methods_names[mi]} exception:{e}, {cur_idx}")
                continue


            # saving results for the method
            method_params = ' '.join([f"({k};{v})" for k, v in m._get_params().items()])

            predictions_through_time = [str(final_pred_history[t]) if t < len(final_pred_history) else "" for t in range(max_length)]

            predictions_through_time = ', '.join(predictions_through_time)
            logger.info(f"{my_stream.__name__}, {method_params}, {B},{FREQUENCY_OF_PREDICTIONS},{metric_fun.__name__}, {', '.join([str(t.get())for t in metrics])},{predictions_through_time}")
            results[methods_names[mi]][metric_fun.__name__] = metrics, final_pred_history

    return my_stream.__name__, results




def train_and_evaluate(initial_stream, Q, probas, methods, methods_params, methods_names, 
                       metric_funs,delay_type, K, B, delay, logger,warm_up_period=10, make_intial_lfs = False):
    '''
    Main function of the experiment 

    Parameters
    ----------
    initial_stream: list
        List of tuples (int,int,dict,int)

    Q: list 
        List of stream boundries (sections of the stream)

    probas: list
        List of probabilites with which the labels are removed

    methods: list
        List of classification methods

    methods_params: list
        List of dictionaries with methods parameters

    methods_names: list
        List of names of the methods

    metric_funs: list
        List of metrics

    delay_type: float
        The proabaility of delay

    K: int
        frequency of predictions

    B: int
        number of bins
    logger:
        logger

    warm_up_period: int
        The amount of the first instances, which should not be delayed
    
    make_intial_lfs: bool
        Indictates if from the initial stream the unlabelled instances should be removed


    Returns
    --------
    
    results:
        results of the experiment

    '''


    pool = ThreadPool(NUMBER_OF_THREADS)
    results = {}
    for q in Q:        
        current_stream = StreamSection( f'{initial_stream.__name__}_{q[0]}_{q[1]}',initial_stream.stream[q[0]:q[1]], initial_stream.is_fully_labelled)
        # preparing streams part
        stream_set = generate_streams(
            current_stream,current_stream.__name__ , q, probas, delay_type, delay, warm_up_period,make_intial_lfs,logger)
        logger.debug('Streams generated')
        # train and evaluation part
        max_length = int(
            np.ceil(max([len(initial_stream.stream) for st in stream_set])/FREQUENCY_OF_PREDICTIONS))
        results_for_q = pool.map(lambda my_stream: train_for_stream(my_stream, methods, methods_params, methods_names,
                                                                    metric_funs, K, B, warm_up_period, max_length,logger), stream_set)

        results[q] = results_for_q
    return results
