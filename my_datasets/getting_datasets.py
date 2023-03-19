
import river.datasets as datasets
from river import stream
from constants import RANDOM_SEED
import pandas as pd
import os

# methods for getting datasets


# real data sets 
# TOD): zmienic sciezki


def get_Airlines() -> list:
    '''
    returns Airlines data set #TODO maybe add one hot encoding - or some methods will not be applicable
    '''
    return [(dict((i, x[i]) for i in x.keys() if i != 'Delay'), x['Delay'])
            for x, _ in stream.iter_arff('C:\\Users\\gosia\\Desktop\\studia\\magisterka\\env\\mgr\\my_datasets\\airlines.arff.zip', compression='infer')]


def get_Electricity() -> list:
    '''
    returns Electricity  data set #TODO maybe add one hot encoding - or some methods will not be applicable
    UP mapped to 1
    DOWN mapped to 0 
    '''
    return [(dict((i, x[i]) for i in x.keys() if i != 'class'), x['class'] == 'UP')
            for x, _ in stream.iter_arff('C:\\Users\\gosia\\Desktop\\studia\\magisterka\\env\\mgr\\my_datasets\\elecNormNew.arff.zip', compression='infer')]


def get_CoverType() -> list:
    '''
    returns CoverType data set 
    '''
    cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
            'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    cols.extend([f'Wilderness_Area_{i}' for i in range(4)])
    cols.extend([f'Soil_Type_{i}' for i in range(40)])
    cols.append('Cover_Type')

    data_types = [float for i in range(10)]
    data_types.extend([bool for i in range(44)])
    data_types.append(int)

    mappings = dict(zip(cols, data_types))

    df = pd.read_csv('C:\\Users\\gosia\\Desktop\\studia\\magisterka\\env\\mgr\\my_datasets\\covtype.data.gz', compression='infer',
                     header=None, names=cols, dtype=mappings)
    cover_type = [(x, y) for x, y in zip(
        df.drop(columns='Cover_Type').to_dict('records'), df['Cover_Type'])]
    return cover_type


# synthetic data streams

def get_LED(noise_percentage: float = 0, irrelevant_features: bool = False, n_drift_features: int = 0):
    '''
    returns LEDDrift data stream -> the parameters are the same as in datasets.synth.LEDDrift class
    '''
    led = datasets.synth.LEDDrift(seed=RANDOM_SEED, noise_percentage=noise_percentage,
                                  irrelevant_features=irrelevant_features, n_drift_features=n_drift_features)
    return led


def get_Hyperplane(n_features: int = 10, n_drift_features: int = 2, mag_change: float = 0,
                   noise_percentage: float = 0.05, sigma: float = 0.1):
    '''
    returns Hyperplane data stream -> parameters the same as in datasets.synth.Hyperplane
    '''
    hp = datasets.synth.Hyperplane(seed=RANDOM_SEED, n_features=n_features, n_drift_features=n_drift_features,
                                   mag_change=mag_change, noise_percentage=noise_percentage, sigma=sigma)
    return hp


def get_AGRAWL(classification_function: int = 0,  balance_classes: bool = False, perturbation: float = 0):
    '''
    returns the AGRAWL data stream -> parameters the same as in datasets.synth.Agrawl
    '''
    agrawl = datasets.synth.Agrawal(classification_function=classification_function, seed=RANDOM_SEED, balance_classes=balance_classes,
                                   perturbation=perturbation)
    return agrawl


def get_RandomRBF(n_classes: int = 2, n_features: int = 10, n_centroids: int = 50):
    '''
    returns the RandomRBF data stream -> parameters same as in datasets.synth.RandomRBFDrift
    '''
    rrbf = datasets.synth.RandomRBFDrift(seed_model=RANDOM_SEED, seed_sample=RANDOM_SEED,
                                         n_classes=n_classes, n_features=n_features, n_centroids=n_centroids)
    return rrbf
