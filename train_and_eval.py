from streams.utils import FL, FU, generate_stream_section
from streams.stream_section import StreamSection
from river import metrics
from multiprocessing.pool import ThreadPool
from constants import NUMBER_OF_THREADS


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
        elif b == B+1:
            metrics[b].update(true_label, preds[-1])
        else:
            y_pred = get_most_frequent(
                preds[int(1+interval*(b-1)):int((1+interval*b))])
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


def add_delay(stream, delay, no_delete_period, dataset_name, q):
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


def generate_streams(datastream, dataset_name, q, probas, with_delay, delay, warm_up_period):
    stream_set = []  # TODO chcek where this should be placed
    initial_stream = generate_stream_section(
        datastream, f'{dataset_name}_init_{q[0]}_{q[1]}', dataset_name, q[0], q[1])
    stream_set.append(initial_stream)
    if with_delay:
        initial_stream = add_delay(
            initial_stream.stream, delay, warm_up_period, dataset_name, q)
        dataset_name += '_delay'
        stream_set.append(initial_stream)

    for p in probas:
        ssl_stream = StreamSection(f'{dataset_name}_ssl_{p}_{q[0]}_{q[1]}', FU(
            initial_stream.stream, p, warm_up_period), False)
        lfs_stream = StreamSection(
            f'{dataset_name}_lfs_{p}_{q[0]}_{q[1]}', FL(ssl_stream.stream), True)
        stream_set.append(ssl_stream)
        stream_set.append(lfs_stream)
    return stream_set



def train_for_stream( my_stream,methods, methods_params,
                       metric_fun, K, B,warm_up_period):
    results = {}
    print(my_stream.__name__)
    for mi, method in enumerate(methods):
            # initilaze method and variables
        m = method(**methods_params[mi])
        metrics = [metric_fun() for _ in range(B+2)]
        h = m
        L = {}
        P = {}

        for cur_idx, init_idx, x, y in my_stream.stream:
            # TODO: can it be in this place
            make_prediction_for_awaiting(h, cur_idx, P, L, K)
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
                if cur_idx != init_idx and init_idx in P.keys():  # delayed label
                    P[init_idx][cur_idx] = h.predict_one(x)
                    L.pop(init_idx)

                    update_performance_measures(
                        P[init_idx], y, B, metrics)
                # TODO: probably need to implment a better option of evaluation-> for now test then train is used
                else:
                    if m._timestamp > warm_up_period:  # if in warmup period the prediction cannot be made
                        # if it was not delayed only last prediction exists
                        metrics[B+1].update(y, h.predict_one(x))

                h = h.learn_one(x, y)
        results[mi] = metrics
    return my_stream.__name__,results

# K how many new labelled instances need to arrive before the new prediction is made
def train_and_evaluate(datastream, dataset_name, Q, probas, methods, methods_params,
                       metric_fun, with_delay, K, B, delay, warm_up_period=10):
    '''
    Main evaluation and traing function
    '''
    pool = ThreadPool(NUMBER_OF_THREADS)
    results = {}
    for q in Q:
        # preparing streams part
        stream_set = generate_streams(
            datastream, dataset_name, q, probas, with_delay, delay, warm_up_period)

        # train and evaluation part
        results_for_q = pool.map(lambda my_stream: train_for_stream( my_stream,methods, methods_params,
                       metric_fun, K, B,warm_up_period),stream_set )
    
            
        results[q] = results_for_q
    return results

        
