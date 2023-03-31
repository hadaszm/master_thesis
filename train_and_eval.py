from streams.utils import FL, FU, generate_stream_section
from streams.stream_section import StreamSection
from river import metrics
from multiprocessing.pool import ThreadPool
from constants import NUMBER_OF_THREADS
import numpy as np
import logging
import datetime

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
        Number of bins #TODO add somewhere the describtion

    metrics: dict
        The dictionary holding the metrics to calculate the results
    '''
    interval = (len(predictions)-1)/B
    preds = list(predictions.values())
    for b in range(B+2):
        if b == 0:
            metrics[b].update(true_label, preds[0])
        if b == B+1:
            metrics[b].update(true_label, preds[-1])
        if preds[int(1+interval*(b)):int((1+interval*(b+1)))]: #interval not empty
                y_pred = get_most_frequent(
                    preds[int(1+interval*(b)):int((1+interval*(b+1)))])
                metrics[b].update(true_label, y_pred)


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
    return StreamSection(f'{dataset_name}_delay_{q[0]}_{q[1]}', new_stream, False)

def add_delay_random(stream, max_delay, no_delete_period, dataset_name, q):
    """ Add random delay (1, max_delay)"""
    new_stream = []
    used_indexes = []
    i = 0
    for idx_1, idx_2, x, y, in stream:
        if i < no_delete_period:
            new_stream.append((i, i, x, y))
            i = i+1
            continue
        while i in used_indexes:
            i+=1
        delay = np.random.randint(1,max_delay)
        while i+delay in used_indexes:
            delay+=1 # if sampling again infinite loop possible
        new_stream.append((i, i, x, None))
        new_stream.append((i+delay, i, x, y))
        used_indexes.extend([i,i+delay])
        i += 1
    new_stream.sort(key=lambda x: x[0])
    return StreamSection(f'{dataset_name}_delay_{q[0]}_{q[1]}', new_stream, False)


def generate_streams(initial_stream, dataset_name, q, probas, delay_type, delay, warm_up_period):
    stream_set = []  # TODO chcek where this should be placed

    stream_set.append(initial_stream)
    if delay_type == 1:
        initial_stream = add_delay_constant(
            initial_stream.stream, delay, warm_up_period, dataset_name, q)
        dataset_name += '_constant_delay'
    elif delay_type == 2:
        initial_stream = add_delay_random(
            initial_stream.stream, delay, warm_up_period, dataset_name, q)
        dataset_name += '_random_delay'
        stream_set.append(initial_stream)

    for p in probas:
        ssl_stream = StreamSection(f'{dataset_name}_ssl_{p}_{q[0]}_{q[1]}', FU(
            initial_stream.stream, p, warm_up_period), False)
        lfs_stream = StreamSection(
            f'{dataset_name}_lfs_{p}_{q[0]}_{q[1]}', FL(ssl_stream.stream), True)
        stream_set.append(ssl_stream)
        stream_set.append(lfs_stream)
    return stream_set



def train_for_stream( my_stream,methods, methods_params,methods_name,
                       metric_fun, K, B,warm_up_period):
    results = {}
    # how many labelled instances appered -> needed for prediction of awaiting examples 
    labelled_insances_cnt = 0
    logging.debug(f" Start processing {my_stream.__name__}")
    for mi, method in enumerate(methods):
            # initilaze method and variables
        m = method(**methods_params[mi])
        metrics = [metric_fun() for _ in range(B+2)]
        h = m
        L = {}
        P = {}
        logging.debug(f" Start processing method {methods_name[mi]}")
        for cur_idx, init_idx, x, y in my_stream.stream:
            # TODO: can it be in this place
           
            # unlabelled instance
            if y is None:
                # add instnace and index
                L[cur_idx] = x
                P[cur_idx] = {}
                P[cur_idx][cur_idx] = h.predict_one(x)
                # TODO: think what to do if the method cannot deal with unlabelled
                h = h.learn_one(x)

            # labelled instance
            else:
                labelled_insances_cnt+=1
                if cur_idx != init_idx and init_idx in P.keys():  # delayed label
                    P[init_idx][cur_idx] = h.predict_one(x)
                    L.pop(init_idx)

                    update_performance_measures(
                        P[init_idx], y, B, metrics)
                # TODO: probably need to implment a better option of evaluation-> for now test then train is used
                else: #TODO: dopytac sie czy tojest ok 
                    if m._timestamp > warm_up_period:  # if in warmup period the prediction cannot be made
                        # if it was not delayed only last prediction exists
                        metrics[B+1].update(y, h.predict_one(x))
                
                make_prediction_for_awaiting(h, labelled_insances_cnt, P, L, K) 
                h = h.learn_one(x, y)
        method_params = ' '.join([f"({k};{v})" for k,v in m._get_params().items()])
        logging.info(f"{my_stream.__name__}, {method_params}, {', '.join([str(t.get())for t in metrics])}")        
        results[methods_name[mi]] = metrics
    return my_stream.__name__,results

# K how many new labelled instances need to arrive before the new prediction is made
def train_and_evaluate(initial_stream, dataset_name, Q, probas, methods, methods_params,methods_name,
                       metric_fun, delay_type, K, B, delay,warm_up_period=10):
    '''
    The initial stream section needs to passed
    Main evaluation and traing function
    delay_type - 0 - NONE, 1 - equal, 2- random
    '''

    now = datetime.datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    logging.basicConfig(filename=f'logs\\{date_time}.log', filemode='w', format='%(asctime)s - %(message)s',level=logging.INFO,datefmt='%d-%b-%y %H:%M:%S')
    pool = ThreadPool(NUMBER_OF_THREADS)
    results = {}
    for q in Q:
        # preparing streams part
        stream_set = generate_streams(
            initial_stream, dataset_name, q, probas, delay_type, delay,warm_up_period)
        logging.debug('Streams generated')
        # train and evaluation part
        results_for_q = pool.map(lambda my_stream: train_for_stream( my_stream,methods, methods_params, methods_name,
                       metric_fun, K, B,warm_up_period),stream_set )
    
            
        results[q] = results_for_q
    return results

        
