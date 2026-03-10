from common import *
import logging
logger = logging.getLogger(name = __name__)
import platform
from psutil import cpu_count
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tqdm.keras import TqdmCallback
import tensorflow as tf
from tensorflow.keras import layers, optimizers

from ..processor import Processor

#threads
cpus = cpu_count(logical = False) #physical
threads = cpus - 1
if platform.system() == 'Linux':
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    if 'GenuineIntel' in cpuinfo: #server
        threads = 2
elif platform.system() == 'Darwin': #local
    threads = max(threads - 4, 1)
logger.info(f"#threads: {threads}")

tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)


class Autoencoder:
    def __init__(self):
        self._processor = Processor()
        self._scaler = MinMaxScaler(feature_range = (0, 1))

    def __repr__(self)->str:
        return 'autoencoder(in progress)'
    def get_name(self)->str:
        return 'Autoencoder'

    def fit(self, _X:pl.DataFrame, categorical:list[str] = [], other:str = '..other', random_seed:int|None = None, verbose:int = 0)->None:
        assert isinstance(_X, pl.DataFrame), 'wrong type'
        assert isinstance(categorical, list), 'wrong type'
        assert isinstance(other, str), 'wrong type'
        assert isinstance(random_seed, int|None), 'wrong type'
        assert isinstance(verbose, int), 'wrong type'
        X = _X.clone(); del _X

        #processor fit-transform
        self._processor.fit(X, categorical, other)
        X = self._processor.transform(X, onehot = True)
        logger.info(f"Training Size: {X.shape}")

        #scaler fit-transform
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        dims = [256, 128, 64, 32, 16, 8]
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()

        #layers
        for i in dims[:-1]: #encoder
            self.encoder.add(layers.Dense(i, activation = 'gelu', kernel_initializer = 'he_uniform'))
        self.encoder.add(layers.Dense(dims[-1], activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
        for i in dims[-2::-1]: #decoder
            self.decoder.add(layers.Dense(i, activation = 'gelu', kernel_initializer = 'he_uniform'))
        self.decoder.add(layers.Dense(X.shape[1], activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))

        #model
        self.model = tf.keras.Sequential([
            self.encoder,
            self.decoder,
            ])
        optimizer = optimizers.AdamW(
            learning_rate = 0.0001,
            epsilon = 1e-6,
            )

        #fitted
        self.model.compile(
            optimizer = optimizer,
            loss = 'mean_squared_error',
            steps_per_execution = 100+1,
            )
        self.model.fit(
            x = X_scaled,
            y = X_scaled,
            batch_size = len(X) // 100,
            epochs = 100,
            verbose = 0,
            callbacks = [TqdmCallback(verbose = verbose)],
            )


        # - range-normalized error -

        #reconstructed
        reconstructed = self.model.predict_on_batch(X_scaled)
        reconstructed = self._scaler.inverse_transform(reconstructed)

        #scale
        self._trainscale = X.max(axis = 0) - X.min(axis = 0)
        self._trainscale[self._trainscale == 0] = 1

        #scores
        error = (reconstructed - X) / np.expand_dims(self._trainscale, 0)
        self._trainscore = np.absolute(error).mean(axis = 1, dtype = 'float64')


    def numerize(self, df:pl.DataFrame)->np.ndarray:
        assert isinstance(df, pl.DataFrame), 'wrong type'
        return self._processor.transform(df, onehot = True)
    def numeric_reconstruct(self, _arr:pl.DataFrame)->np.ndarray:
        assert isinstance(_arr, pl.DataFrame), 'wrong type'
        arr = _arr.clone(); del _arr

        #prepared
        arr = self._processor.transform(arr, onehot = True)
        arr = self._scaler.transform(arr)

        #reconstructed
        arr_re = self.model.predict_on_batch(arr)
        arr_re = self._scaler.inverse_transform(arr_re)

        return arr_re


    def predict(
        self,
        df:pl.DataFrame,
        train_fpr:float|int = 0.0001,
        return_ranks:bool = False,
        return_scores:bool = False,
        ):
        assert isinstance(df, pl.DataFrame), 'wrong type'
        assert isinstance(train_fpr, float|int), 'wrong type'
        assert isinstance(return_ranks, bool), 'wrong type'
        assert isinstance(return_scores, bool), 'wrong type'
        returns = []

        #reconstructed
        ori = self.numerize(df)
        rec = self.numeric_reconstruct(df)

        #scores
        error = (rec - ori) / np.expand_dims(self._trainscale, 0)
        score = np.absolute(error).mean(axis = 1, dtype = 'float64')

        #prediction
        threshold = np.quantile(self._trainscore, 1 - train_fpr, axis = 0).tolist()
        prediction = score >= threshold

        #ranking
        rank = stats.rankdata(score, method = 'min', axis = 0)

        returns.append(prediction)
        if return_ranks:
            returns.append(rank)
        if return_scores:
            returns.append(score)

        if len(returns) == 0:
            pass
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns
