from common import *
import logging
logger = logging.getLogger(name = __name__)
import platform
from psutil import cpu_count
from random import randint
from scipy import stats

from ..processor import Processor

from pyod.models.auto_encoder import AutoEncoder as AE

n_jobs = 'N/A'
logger.info(f"#threads: {n_jobs}")


class Autoencoder:
    def __init__(self):
        self._processor = Processor()

    def __repr__(self)->str:
        return 'autoencoder'
    def get_name(self)->str:
        return 'Autoencoder'

    def fit(
        self,
        param_X:pl.DataFrame,
        categorical:list = [],
        other:str = '..other',
        hidden_layers:list = [256, 128, 64, 32, 16, 8],
        dropout:float|int = 0.01,
        verbose:int = 1,
        random_seed:int|None = None,
        )->None:
        X = param_X.clone(); del param_X
        if random_seed is None:
            random_seed = randint(0, 255) #pyod default not random

        #processor
        self._processor.fit(X, categorical, other)

        #processed
        X = self._processor.transform(X, onehot = True)
        logger.info(f'Training Size: {X.shape}')

        self._model = AE(
            contamination = 0.0001,
            preprocessing = True,
            hidden_neuron_list = hidden_layers,
            batch_size = max(len(X) // 100, 32),
            epoch_num = 100,
            verbose = verbose,
            random_state = random_seed,
            dropout_rate = dropout,
            )
        self._model.fit(X)

        #trainscore stored
        self._trainscore = self._model.decision_function(X, batch_size = len(X) // 100).astype('float64')


    def predict(
        self,
        param_data:pl.DataFrame,
        train_fpr:float|int = 0.0001,
        return_ranks:bool = False,
        return_scores:bool = False,
        ):
        data = param_data.clone(); del param_data
        returns = []

        #processed
        data = self._processor.transform(data, onehot = True)

        #threshold
        threshold = np.quantile(self._trainscore, 1 - train_fpr, axis = 0).tolist()

        #predicted
        score = self._model.decision_function(data).astype('float64')
        prediction = score >= threshold
        returns.append(prediction)

        #ranking
        if return_ranks:
            rank = stats.rankdata(score, method = 'min', axis = 0)
            returns.append(rank)

        #anomaly scores
        if return_scores:
            returns.append(score)

        if len(returns) == 0:
            return None
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns
