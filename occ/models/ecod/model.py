from common import *
import logging
logger = logging.getLogger(name = __name__)
import platform
from psutil import cpu_count
from scipy import stats

from ..processor import Processor

from pyod.models.ecod import ECOD as _ECOD

#threads
cpus = cpu_count(logical = False) #physical
n_jobs = cpus - 1
if platform.system() == 'Linux':
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    if 'GenuineIntel' in cpuinfo: #server
        n_jobs = 2
elif platform.system() == 'Darwin': #local
    n_jobs = 1 # The parallel implementation does not share memory, resulting in OOM.
logger.info(f"#threads: {n_jobs}")


class ECOD:
    def __init__(self):
        self._processor = Processor()

    def __repr__(self):
        return 'ecod'
    def get_name(self):
        return 'ECOD'

    def fit(self, param_X, categorical = [], other = '..other', verbose = 1, random_seed = None):
        if not isinstance(param_X, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(categorical, list):
            raise TypeError('\'categorical\' should be a list.')
        if not isinstance(other, str):
            raise TypeError('\'other\' should be a string.')
        if not isinstance(verbose, int):
            raise TypeError('\'verbose\' should be an integer.')
        if not isinstance(random_seed, int) and random_seed is not None:
            raise TypeError('\'random_seed\' should be an integer.')
        X = param_X.clone(); del param_X

        #processor fit
        self._processor.fit(X, categorical, other)

        #processed
        X = self._processor.transform(X, onehot = True)
        logger.info(f'Training Size: {X.shape}')

        self._model = _ECOD(contamination = 0.0001, n_jobs = n_jobs)
        self._model.fit(X)

        #trainscore stored
        self._trainscore = self._model.decision_function(X)


    def predict(self, param_data, train_fpr = 0.0001, return_ranks = False, return_scores = False):
        if not isinstance(param_data, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(train_fpr, float):
            raise TypeError('\'train_fpr\' should be a float.')
        if not isinstance(return_ranks, bool):
            raise TypeError('\'return_ranks\' should be boolean.')
        if not isinstance(return_scores, bool):
            raise TypeError('\'return_scores\' should be boolean.')
        data = param_data.clone(); del param_data
        returns = []

        #processed
        data = self._processor.transform(data, onehot = True)

        #threshold
        threshold = np.quantile(self._trainscore, 1 - train_fpr, axis = 0)

        #predicted
        score = self._model.decision_function(data)
        prediction = score >= threshold
        returns.append(prediction)

        #ranking
        if return_ranks:
            rank = stats.rankdata(score, method = 'min', axis = 0)
            returns.append(rank)
        if return_scores:
            returns.append(score)

        if len(returns) == 0:
            return None
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns
