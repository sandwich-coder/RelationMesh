from common import *
import platform
from psutil import cpu_count
import logging
logger = logging.getLogger(name = __name__)
from scipy import stats

from ..processor import Processor

from pyod.models.ocsvm import OCSVM as _OCSVM

n_jobs = 'N/A'
logger.info(f"#threads: {n_jobs}")


class OCSVM:
    def __init__(self):
        self._processor = Processor(onehot = True)

    def __repr__(self)->str:
        return 'ocsvm'
    def get_name(self)->str:
        return 'OCSVM'

    # Random seed is for signature consistency. Has no effect in this model.
    def fit(self, param_X:pl.DataFrame, categorical:list = [], other:str = '..other', verbose:bool = True, random_seed:int|None = None)->None:
        X = param_X.clone(); del param_X

        #processor fit
        self._processor.fit(X, categorical, other)

        #processed
        X = self._processor.transform(X)
        logger.info(f'Training Size: {X.shape}')

        self._model = _OCSVM(
            contamination = 0.0001,
            verbose = verbose,
            )
        self._model.fit(X)

        #trainscore stored
        self._trainscore = self._model.decision_function(X)


    def predict(self, param_data:pl.DataFrame, train_fpr:float|int = 0.0001, return_ranks:bool = False, return_scores:bool = False):
        data = param_data.clone(); del param_data
        returns = []

        #processed
        data = self._processor.transform(data)

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
