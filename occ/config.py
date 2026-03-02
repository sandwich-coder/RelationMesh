import pathlib
from sklearn.preprocessing import StandardScaler


__all__ = [

    'DATASET',
    'ANOMALY_TYPE',
    'RANDOM_SEED',
    'MAX_P_ANOMALOUS',

    'TRAIN_FPR',

    'RESULT_DIR',

    'boosters',

    ]


#data
DATASET = 'cicids2017'
ANOMALY_TYPE = None
RANDOM_SEED = 1
MAX_P_ANOMALOUS = None

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
