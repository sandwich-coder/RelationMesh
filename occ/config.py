import pathlib
from sklearn.preprocessing import StandardScaler

from models import (

    RelationMesh,
    IsolationForest,
    Autoencoder,
    ECOD,

    #in progress
    Autoencoder_,
    )


__all__ = [

    'DATASET',
    'ANOMALY_TYPE',
    'RANDOM_SEED',
    'MAX_P_ANOMALOUS',

    'models',

    'TRAIN_FPR',

    'RESULT_DIR',

    'boosters',

    ]


#data
DATASET = 'nsl-kdd'
ANOMALY_TYPE = None
RANDOM_SEED = 1
MAX_P_ANOMALOUS = None

#models
models = [
    RelationMesh(base_learner = 'rf'),
    IsolationForest(),
    Autoencoder(),
    Autoencoder_(),
    ECOD(),
    ]

#threshold
TRAIN_FPR = 0.01

#result
RESULT_DIR = pathlib.Path('temp')



" ========== booster settings ========== "


boosters = {

#$    'cicids2017': {
#$
#$        'Scaler': StandardScaler,
#$        'regressor': {
#$            'max_delta_step': 0,
#$            'objective': 'reg:squarederror',
#$            'eval_metric': 'rmse',
#$            },
#$
#$        },

#$    'unsw-nb15': {
#$
#$        'Scaler': StandardScaler,
#$        'regressor': {
#$            'max_delta_step': 0,
#$            'objective': 'reg:squarederror',
#$            'eval_metric': 'rmse',
#$            },
#$
#$        },

#$    'unsw-nb15/small': {
#$
#$        'Scaler': StandardScaler,
#$        'regressor': {
#$            'max_delta_step': 0,
#$            'objective': 'reg:squarederror',
#$            'eval_metric': 'rmse',
#$            },
#$
#$        },

#$    'nsl-kdd': {
#$
#$        'Scaler': StandardScaler,
#$        'regressor': {
#$            'max_delta_step': 0,
#$            'objective': 'reg:squarederror',
#$            'eval_metric': 'rmse',
#$            },
#$
#$        },

    }
