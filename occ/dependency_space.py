if __name__ != '__main__':
    raise RuntimeError('This module is a standalone experiment.')
from rich.traceback import install as rich_traceback
rich_traceback()

from common import *
import logging
logger = logging.getLogger()
logging.basicConfig(level = 'INFO')
from datetime import datetime as Datetime

from loader import Loader
from models import (
    RelationMesh,
    IsolationForest,
    OCSVM,
    ECOD,
    Autoencoder,
    )
from evaluate import fast_result, aucplot, scoreplot

dataset = 'cicids2017'
anomaly_type = 'reconnaissance'
random_seed = 1
train_fpr = 0.01

#starting log
logger.info(f"Dataset: {dataset}")
logger.info(f"Anomaly Type: {anomaly_type}")
logger.info(f"Random Seed: {random_seed}")
logger.info(f"Train-fpr: {train_fpr}")



# - data -

loader = Loader()
normal_t, normal_v = loader.load(dataset, split_seed = random_seed)
anomalous_t, anomalous_v = loader.load(
    dataset,
    anomalous = True,
    anomaly_type = anomaly_type,
    split_seed = random_seed,
    )


# - models -

rmesh = RelationMesh(base_learner = 'rf')

baselines = [
    IsolationForest(),
    Autoencoder(),
    ECOD(),
    ]

baselines_ = [
    IsolationForest(),
    Autoencoder(),
    ECOD(),
    ]


# - trained -

X = normal_t.clone()

rmesh.fit(X, categorical = loader.get_catcols(dataset), random_seed = random_seed)

for i in baselines:
    i.fit(X, categorical = loader.get_catcols(dataset), random_seed = random_seed)

for i in baselines_:
    i.fit(
        rmesh.predict(X, return_residuals = True)[1],
        random_seed = random_seed,
        )


# - evaluation -

#prepared
label_t = np.array([False] * len(normal_t) + [True] * len(anomalous_t), dtype = 'bool')
data_t = pl.concat([normal_t, anomalous_t], how = 'vertical')
label_v = np.array([False] * len(normal_v) + [True] * len(anomalous_v), dtype = 'bool')
data_v = pl.concat([normal_v, anomalous_v], how = 'vertical')

#predicted (training)
predictions_t, ranks_t, scores_t = {}, {}, {}
predictions_t[rmesh.get_name()], ranks_t[rmesh.get_name()], scores_t[rmesh.get_name()] = rmesh.predict(
    data_t,
    train_fpr = train_fpr,
    return_ranks = True,
    return_scores = True,
    )
for i in baselines:
    predictions_t[i.get_name()], ranks_t[i.get_name()], scores_t[i.get_name()] = i.predict(
        data_t,
        train_fpr = train_fpr,
        return_ranks = True,
        return_scores = True,
        )
for i in baselines_:
    predictions_t[i.get_name() + '✳︎'], ranks_t[i.get_name() + '✳︎'], scores_t[i.get_name() + '✳︎'] = i.predict(
        rmesh.predict(data_t, return_residuals = True)[1],
        train_fpr = train_fpr,
        return_ranks = True,
        return_scores = True,
        )

#predicted
predictions_v, ranks_v, scores_v = {}, {}, {}
predictions_v[rmesh.get_name()], ranks_v[rmesh.get_name()], scores_v[rmesh.get_name()] = rmesh.predict(
    data_v,
    train_fpr = train_fpr,
    return_ranks = True,
    return_scores = True,
    )
for i in baselines:
    predictions_v[i.get_name()], ranks_v[i.get_name()], scores_v[i.get_name()] = i.predict(
        data_v,
        train_fpr = train_fpr,
        return_ranks = True,
        return_scores = True,
        )
for i in baselines_:
    predictions_v[i.get_name() + '✳︎'], ranks_v[i.get_name() + '✳︎'], scores_v[i.get_name() + '✳︎'] = i.predict(
        rmesh.predict(data_v, return_residuals = True)[1],
        train_fpr = train_fpr,
        return_ranks = True,
        return_scores = True,
        )

#training
print('\n - Training - \n')
fast_result(label_t, predictions_t)
aucplot(label_t, ranks_t, save_plots = True, prefix = f"{dataset}-training-", timestamp = True)
scoreplot(label_t, scores_t, save_plots = True, prefix = f"{dataset}-training-", timestamp = True)

#validation
print('\n - Validation - \n')
fast_result(label_v, predictions_v)
aucplot(label_v, ranks_v, save_plots = True, prefix = f"{dataset}-", timestamp = True)
scoreplot(label_v, scores_v, save_plots = True, prefix = f"{dataset}-", timestamp = True)
