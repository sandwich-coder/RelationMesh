from common import *
import logging
logger = logging.getLogger(name = __name__)
from scipy import stats
from sklearn.ensemble import IsolationForest as IF
import platform
from psutil import cpu_count

from ..processor import Processor

#threads
cpus = cpu_count(logical = False) #physical
n_jobs = cpus - 1
if platform.system() == 'Linux':
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    if 'GenuineIntel' in cpuinfo: #server
        n_jobs = 2
elif platform.system() == 'Darwin': #local
    n_jobs = max(n_jobs - 4, 1)
logger.info(f"#threads: {n_jobs}")


class IsolationForest:
    def __init__(self):
        self._processor = Processor()

    def __repr__(self):
        return 'isolation-forest'
    def get_name(self):
        return 'IsolationForest'

    def fit(self, param_X, categorical = [], other = '..other', verbose = 1, random_seed = None):
        if not isinstance(param_X, pl.DataFrame):
            raise TypeError('Input should be a \'polars.DataFrame\'.')
        if not isinstance(categorical, list):
            raise TypeError('\'categorical\' should be a list.')
        if not isinstance(other, str):
            raise TypeError('\'other\' should be a string.')
        if not isinstance(verbose, int):
            raise TypeError('\'verbose\' should be an integer.')
        if not isinstance(random_seed, int) and random_seed is not None:
            raise TypeError('\'random_seed\' should be an integer.')
        X = param_X.clone(); del param_X

        self._processor.fit(X, categorical, other)

        X = self._processor.transform(X, onehot = True)
        logger.info(f'Training Size: {X.shape}')

        self._model = IF(
            n_estimators = 300,
            n_jobs = n_jobs,
            contamination = 0.0001,
            verbose = verbose,
            random_state = random_seed,
            )

        self._model.fit(X)
        self._trainscore = -self._model.decision_function(X)


    def predict(self, param_data, train_fpr = 0.0001, return_ranks = False, return_scores = False):
        if not isinstance(param_data, pl.DataFrame):
            raise TypeError('Input should be a \'polars.DataFrame\'.')
        if not isinstance(train_fpr, (float, int)):
            raise TypeError('\'train_fpr\' should be a number.')
        if not isinstance(return_ranks, bool):
            raise TypeError('\'return_ranks\' should be boolean.')
        if not isinstance(return_scores, bool):
            raise TypeError('\'return_scores\' should be boolean.')
        data = param_data.clone(); del param_data
        returns = []

        data = self._processor.transform(data, onehot = True)

        threshold = np.quantile(self._trainscore, 1 - train_fpr)

        score = -self._model.decision_function(data)
        prediction = score >= threshold

        returns.append(prediction)
        if return_ranks:
            returns.append(
                stats.rankdata(score, axis = 0, method = 'min'),
                )
        if return_scores:
            returns.append(score)

        if len(returns) == 0:
            return None
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns
