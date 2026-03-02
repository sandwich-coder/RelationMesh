import os, sys

from copy import deepcopy as copy
import types, inspect
import platform
import time
import logging
import numpy as np
from scipy import linalg as la
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams.update({

    'figure.figsize':[10, 10],
    'figure.edgecolor':'none',

    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.facecolor':'none',

    'lines.markersize':1,
    'lines.linewidth':0.5,
    'scatter.edgecolors':'none',

    'legend.facecolor':'none',
    'legend.edgecolor':'none',

    })

import polars as pl
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import torch
from torch import optim, nn
from rich import print
from rich.console import Console

from numpy.random import default_rng
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import pandas as pd


#random seed
random_seed = 42

#random generator
rng = default_rng(seed = random_seed)

#perturbation
def perturb(arr, width = 0.1):
    noise = rng.random(size = arr.shape, dtype = 'float64')
    noise = (noise - noise.mean(axis = 0, dtype = 'float64', keepdims = True)) * np.float64(width)
    return arr + noise


# - data -

#whole
data = np.linspace(-1, 1, endpoint = True, num = 10000+1, dtype = 'float64')
data = data.reshape([data.shape[0], -1])
label = (np.sin(np.pi * data, dtype = 'float64'))[:, 0]
label = perturb(label) #perturbed
label[3000:3001] = 10 #noise

#train-val split
idx_val = np.zeros(len(data), dtype = 'bool')
idx_val[5000:6000] = True
data_t = data[~idx_val]
label_t = label[~idx_val]
data_v = data[idx_val]
label_v = data[idx_val]


# - model -

X = data_t.copy()
y = label_t.copy()

# THe max delta step is not effective in ignoring 0.01% outliers since every step is the gradient, as expected.
# However, the L2 regularization is not as effective as expected based on the formula, which tells is equivalent to introducing phantom points already fitted.
## It is worth trying high leaves, high learning rate, and low iterations.
config = {

    'booster': 'gbtree',
    'enable_categorical': True,
    'grow_policy': 'lossguide', 'max_depth': 0,
    'max_leaves': max(len(X) // 10000, 10),
    'max_delta_step': 1,

    'objective': 'reg:pseudohubererror',
    'reg_lambda': 10,

    'colsample_bytree': 0.5,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,

    'eval_metric': 'mphe',
    'early_stopping_rounds': 0,

    'tree_method': 'hist',
    'device': 'cpu',
    'n_jobs': 1,

    'learning_rate': 0.01,
    'n_estimators': 1000+1,
    'max_bin': len(X),

    'random_state': random_seed,

    }

scaler = RobustScaler(copy = True)
iforest = IsolationForest(n_estimators = 1000, contamination = 0.0001, verbose = 1)
model = XGBRegressor(**config)

Xy = np.concatenate([X, np.expand_dims(y, 1)], axis = 1)

scaler.fit(y.reshape([y.shape[0], -1]))
iforest.fit(Xy)

y_scaled = scaler.transform(np.expand_dims(y, 1)).squeeze()
weight = iforest.decision_function(Xy)
weight = weight - weight.min(axis = 0)
model.fit(
    X, y_scaled,
    eval_set = [(X, y_scaled)],
    verbose = 100,
    sample_weight = weight,
    )


# - plots -

#y limit of the first graph
ylim = label - label_t.mean(axis = 0, dtype = 'float64', keepdims = True)
ylim = ylim * 1.1
ylim = [ylim.min(axis = 0).tolist(), ylim.max(axis = 0).tolist()]

fig = pp.figure(layout = 'constrained', facecolor = 'white')
fig.suptitle('Newton Boosting')
ax = fig.add_subplot()
ax.set(
    box_aspect = 1,
    )
pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

pred_scaled = model.predict(X).astype('float64')
pred = scaler.inverse_transform(np.expand_dims(pred_scaled, 1)).squeeze()

plot_1 = ax.scatter(
    X[:, 0], y,
    marker = 'o', s = 1, c = 'tab:blue', alpha = 0.7,
    )
plot_2, = ax.plot(
    X[:, 0], pred,
    linestyle  = '-', linewidth = 2, color = 'tab:red', alpha = 1,
    marker = '',
    )
pp.show()
