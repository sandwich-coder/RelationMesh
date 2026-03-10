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

    def __repr__(self)->str:
        return 'isolation-forest'
    def get_name(self)->str:
        return 'IsolationForest'

    def fit(
        self,
        _X:pl.DataFrame,
        categorical:list = [],
        other:str = '..other',
        random_seed:int|None = None,
        verbose:int = 1,
        )->None:
        X = _X.clone(); del _X

        #processor fit-transform
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


    def predict(
        self,
        _data:pl.DataFrame,
        train_fpr:float|int = 0.0001,
        return_ranks:bool = False,
        return_scores:bool = False,
        ):
        data = _data.clone(); del _data
        returns = []

        #prepared
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
