from common import *
import logging
logger = logging.getLogger(name = __name__)
from sklearn.ensemble import (
    RandomForestRegressor as _RandomForestRegressor,
    RandomForestClassifier as _RandomForestClassifier,
    )
from sklearn.preprocessing import OneHotEncoder


class RandomForestRegressor:
    def __init__(self, min_samples_leaf = 1, random_seed = None, n_jobs = 1):
        self._model = _RandomForestRegressor(
            min_samples_leaf = min_samples_leaf,
            random_state = random_seed,
            n_jobs = n_jobs,
            verbose = 1,
            )

    def fit(self, param_X, y, category_columns = []):
        if not isinstance(param_X, pl.DataFrame):
            raise TypeError(f"The input should be a polars.DataFrame. Received {type(param_X)}.")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"The target should be a numpy.ndarray. Received {type(y)}.")
        if not isinstance(category_columns, list):
            raise TypeError('\'category_columns\' should be a list.')
        X = param_X.clone(); del param_X
        self._category_columns = copy(category_columns)

        #onehot
        if len(self._category_columns) < 1:
            pass
        else:

            X_categorical = X.select(self._category_columns)
            X_numerical = X.drop(self._category_columns)

            #fit-transformed and to ndarray
            self._ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore', dtype = 'float64')
            self._ohe.fit(X_categorical)
            X_categorical = self._ohe.transform(X_categorical)
            X_numerical = X_numerical.to_numpy(writable = True)
            X = np.concatenate([X_categorical, X_numerical], axis = 1)


        self._model.fit(X, y)


    def predict(self, param_data):
        if not isinstance(param_data, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        data = param_data.clone(); del param_data

        #onehot
        if len(self._category_columns) < 1:
            pass
        else:

            data_categorical = data.select(self._category_columns)
            data_numerical = data.drop(self._category_columns)

            #onehot transformed and to ndarray
            data_categorical = self._ohe.transform(data_categorical)
            data_numerical = data_numerical.to_numpy(writable = True)
            data = np.concatenate([data_categorical, data_numerical], axis = 1)


        pred = self._model.predict(data)
        return pred



class RandomForestClassifier:
    def __init__(self, min_samples_leaf = 1, random_seed = None, n_jobs = 1):
        self._model = _RandomForestClassifier(
            min_samples_leaf = min_samples_leaf,
            random_state = random_seed,
            n_jobs = n_jobs,
            verbose = 1,
            )

    def __repr__(self):
        return 'random-forest classifier'

    def fit(self, param_X, y, category_columns = []):
        if not isinstance(param_X, pl.DataFrame):
            raise TypeError(f"The input should be a polars.DataFrame. Received {type(param_X)}.")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"The target should be a numpy.ndarray. Received {type(y)}.")
        if not isinstance(category_columns, list):
            raise TypeError('\'category_columns\' should be a list.')
        X = param_X.clone(); del param_X
        self._category_columns = copy(category_columns)

        #onehot
        if len(self._category_columns) < 1:
            pass
        else:

            X_categorical = X.select(self._category_columns)
            X_numerical = X.drop(self._category_columns)

            #fit-transformed and to ndarray
            self._ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore', dtype = 'float64')
            self._ohe.fit(X_categorical)
            X_categorical = self._ohe.transform(X_categorical)
            X_numerical = X_numerical.to_numpy(writable = True)
            X = np.concatenate([X_categorical, X_numerical], axis = 1)


        self._model.fit(X, y)


    def predict_proba(self, param_data):
        if not isinstance(param_data, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        data = param_data.clone(); del param_data

        #onehot
        if len(self._category_columns) < 1:
            pass
        else:

            data_categorical = data.select(self._category_columns)
            data_numerical = data.drop(self._category_columns)

            #onehot transformed and to ndarray
            data_categorical = self._ohe.transform(data_categorical)
            data_numerical = data_numerical.to_numpy(writable = True)
            data = np.concatenate([data_categorical, data_numerical], axis = 1)

        #consistency with xgboost
        self.classes_ = self._model.classes_

        pred_prob = self._model.predict_proba(data)
        return pred_prob
