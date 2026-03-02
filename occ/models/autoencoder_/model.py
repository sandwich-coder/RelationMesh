from common import *
import logging
logger = logging.getLogger(name = __name__)
import platform
from psutil import cpu_count
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, optimizers

from ..processor import Processor


class Autoencoder:
    def __init__(self):
        self._processor = Processor(onehot = True)
        self._scaler = MinMaxScaler(feature_range = (0, 1))

    def __repr__(self):
        return 'autoencoder (in progress)'

    def fit(self, param_X, categorical = [], other = '..other', random_seed = None, verbose = 0):
        if not isinstance(param_X, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(categorical, list):
            raise TypeError('\'categorical\' should be a list.')
        if not isinstance(other, str):
            raise TypeError('\'other\' should be a string.')
        if not isinstance(random_seed, int|None):
            raise TypeError('\'random_seed\' should be an integer.')
        if not isinstance(verbose, int):
            raise TypeError('\'verbose\' should be an integer.')
        X = param_X.clone(); del param_X

        #processor fit-transform
        self._processor.fit(X, categorical, other)
        X = self._processor.transform(X)

        #scaler fit-transform
        self._scaler.fit(X)
        X = self._scaler.transform(X)

        dims = [256, 128, 64, 32, 16, 8]
        self.encoder = tf.Sequential()
        self.decoder = tf.Sequential()

        #encoder
        for i in dims[:-1]:
            self.encoder.add(layers.Dense(i, activation = 'gelu', kernel_initializer = 'he_uniform'))
        self.encoder.add(layers.Dense(dims[-1], activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))

        #decoder
        for i in dims[-2::-1]:
            self.decoder.add(layers.Dense(i, activation = 'gelu', kernel_initializer = 'he_uniform'))
        self.decoder.add(layers.Dense(X.shape[1], activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
