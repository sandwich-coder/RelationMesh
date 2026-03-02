import os, sys
from rich.traceback import install as rich_traceback
rich_traceback()

from common import *
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()
import yaml

from loader import Loader
from models import (
    RelationMesh,
    IsolationForest,
    Autoencoder,
    ECOD,

    #in progress
    Autoencoder_,

    )
from evaluate import fast_result, aucplot, scoreplot, time_start as ts
from config import *

#constants
logger.info(f"Dataset: {DATASET}")
logger.info(f"Anomaly Type: {ANOMALY_TYPE}")
logger.info(f"Random Seed: {RANDOM_SEED}")
logger.info(f"Maximum Anomaly Ratio: {MAX_P_ANOMALOUS}")
logger.info(f"Train fpr: {TRAIN_FPR}")
logger.info(f"Result Directory: {RESULT_DIR}")


# - loaded -

#data
loader = Loader()
normal_t, normal_v = loader.load(DATASET, split_seed = RANDOM_SEED)
if MAX_P_ANOMALOUS is not None:
    anomalous_t, anomalous_v = loader.load(
        DATASET,
        anomalous = True,
        anomaly_type = ANOMALY_TYPE,
        samples = int((len(normal_t) + len(normal_v)) * (MAX_P_ANOMALOUS / (1 - MAX_P_ANOMALOUS))),
        split_seed = RANDOM_SEED,
        sample_seed = RANDOM_SEED,
        )
else:
    anomalous_t, anomalous_v = loader.load(
        DATASET,
        anomalous = True,
        anomaly_type = ANOMALY_TYPE,
        split_seed = RANDOM_SEED,
        )

#models
models = [
    RelationMesh(base_learner = 'default'),
    RelationMesh(base_learner = 'rf'),
    IsolationForest(),
    Autoencoder(),
    ECOD(),
    ]


# - trained -

#training set
X = normal_t.clone()

#train
for i in models:
    if f"{i}" == 'relation-mesh': #my model
        i.fit(
            X,
            categorical = loader.get_catcols(DATASET),
            config = boosters.get(DATASET),
            random_seed = RANDOM_SEED,
            )
    else:
        i.fit(X, categorical = loader.get_catcols(DATASET), random_seed = RANDOM_SEED)


# - evaluation -

#prepared
label_t = np.array([False] * len(normal_t) + [True] * len(anomalous_t), dtype = 'bool')
data_t = pl.concat([normal_t, anomalous_t], how = 'vertical')
label_v = np.array([False] * len(normal_v) + [True] * len(anomalous_v), dtype = 'bool')
data_v = pl.concat([normal_v, anomalous_v], how = 'vertical')

#predicted
predictions_t, ranks_t, scores_t = {}, {}, {}
predictions_v, ranks_v, scores_v = {}, {}, {}
for i in models:

    #training
    temp = i.predict(data_t, train_fpr = TRAIN_FPR, return_ranks = True, return_scores = True)
    predictions_t[f"{i}"] = temp[0]
    ranks_t[f"{i}"] = temp[1]
    scores_t[f"{i}"] = temp[2]

    #validation
    temp = i.predict(data_v, train_fpr = TRAIN_FPR, return_ranks = True, return_scores = True)
    predictions_v[f"{i}"] = temp[0]
    ranks_v[f"{i}"] = temp[1]
    scores_v[f"{i}"] = temp[2]




# - results -

results = {
    'dataset': f"{DATASET}",
    'anomaly-type': f"{ANOMALY_TYPE}",
    'anomalies': '{proportion:.3g}%'.format(
        proportion = (label_v.astype('int64').sum(axis = 0, dtype = 'float64').tolist() / float(len(label_v))) * 100.,
        ),
    'random-seed': RANDOM_SEED,
    'train-fpr': TRAIN_FPR,
    'Booster': boosters.get(DATASET, 'default'),
    'detection-training': {}, 'detection': {},
    'ranking-training': {}, 'ranking': {},
    }

#training
print(f"\n - {DATASET}-training - \n")
results['detection-training'] = fast_result(
    label_t,
    predictions_t,
    return_values = True,
    )
results['ranking-training']  = aucplot(
    label_t,
    ranks_t,
    save_plots = True,
    prefix = f"{DATASET}-training-",
    timestamp = True,
    return_values = True,
    )
scoreplot(
    label_t,
    scores_t,
    save_plots = True,
    prefix = f"{DATASET}-training-",
    timestamp = True,
    )

#validation
print(f"\n - {DATASET} - \n")
results['detection'] = fast_result(
    label_v,
    predictions_v,
    return_values = True,
    )
results['ranking'] = aucplot(
    label_v,
    ranks_v,
    save_plots = True,
    prefix = f"{DATASET}-",
    timestamp = True,
    return_values = True,
    )
scoreplot(
    label_v,
    scores_v,
    save_plots = True,
    prefix = f"{DATASET}-",
    timestamp = True,
    )

#saved
RESULT_DIR.mkdir(parents = True, exist_ok = True)
with open(f"{RESULT_DIR}/{ts}.yaml", 'w') as f:
    yaml.dump(results, f, sort_keys = False)


print('--End--')
