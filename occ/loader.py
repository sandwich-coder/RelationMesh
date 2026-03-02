from common import *
import logging
logger = logging.getLogger(name = __name__)
from sklearn.model_selection import train_test_split

catcols = {
    'nsl-kdd': [
        'protocol',
        'service',
        'flag',
        ],
    'unsw-nb15': [
        'proto',
        'service',
        'state',
        ],
    'unsw-nb15/small': [
        'proto',
        'service',
        'state',
        ],
    'cicids2017': [],
    'higgs': [],
    }


class Loader:
    def __repr__(self):
        return 'loader'

    def __init__(self):
        pass

    def load(
        self,
        name,
        anomalous = False,
        anomaly_type = None,
        p_val = 0.1,
        samples = None,
        split_seed = None,
        sample_seed = None,
        ):
        if not isinstance(name, str):
            raise TypeError('Input should be a string.')
        if not isinstance(anomalous, bool):
            raise TypeError('\'anomalous\' should be boolean.')
        if not isinstance(anomaly_type, str|None):
            raise TypeError('\'anomaly_type\' should be a string.')
        if not isinstance(p_val, float):
            raise TypeError('\'p_val\' should be a float.')
        if not isinstance(samples, int) and samples is not None:
            raise TypeError('\'samples\' should be an integer.')
        if samples is not None:
            assert samples >= 1, 'the number of samples less than 1'
        if not isinstance(split_seed, int) and split_seed is not None:
            raise TypeError('\'split_seed\' should be an integer.')
        if split_seed is not None:
            assert split_seed >= 0, 'the split seed negative'
        if not isinstance(sample_seed, int) and sample_seed is not None:
            raise TypeError('\'sample_seed\' should be an integer.')
        assert 0 < p_val < 1, 'the validation proportion not between 0 and 1'
        if sample_seed is not None:
            assert sample_seed >= 0, 'the sampling seed negative'
        if name[:7] == 'adbench':
            df = np.load(f"datasets/{name}.npz")
            df = np.concatenate([
                df['X'],
                df['y'].reshape([df['y'].shape[0], -1]).astype('float64'),
                ], axis = 1)
            df = pl.from_numpy(
                df,
                schema = ['col' + str(c) for c in range(1, df.shape[1])] + ['label'],
                orient = 'row',
                ).cast({'label': pl.Boolean})
        else:

            df = pl.read_parquet(f"datasets/{name}/data.parquet")

        #selected
        if 'kind' in df.columns:
            if anomaly_type is not None:
                df = df.filter((pl.col('label') == anomalous) & (pl.col('kind') == anomaly_type))
                if len(df) < 1:
                    logger.error(f"There is no instance of the kind \'{anomaly_type}\'.")
            else:
                df = df.filter(pl.col('label') == anomalous)
            df = df.drop(['kind'])
        else:

            df = df.filter(pl.col('label') == anomalous)

        df = df.drop(['label'])

        #sampled
        if samples is not None:
            if samples > len(df):
                logger.warning('The number of samples is greater than the sampled, falling back to no-sample.')
            else:
                df = df.sample(
                    n = samples,
                    shuffle = True,
                    seed = sample_seed,
                    )

        #split
        train, val = train_test_split(
            df,
            test_size = p_val,
            shuffle = True,
            random_state = split_seed,
            )

        return train, val


    def get_catcols(self, name):
        if not isinstance(name, str):
            raise TypeError('Input should be a string.')
        if name[:7] == 'adbench':
            return []
        else:

            return copy(catcols[name])
