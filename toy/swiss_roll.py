import sys, os, subprocess

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
    'figure.facecolor':'none',
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.facecolor':'none',
    'lines.markersize':1,
    'lines.linewidth':0.5,
    'legend.edgecolor':'none',
    'legend.facecolor':'none',
    })

import polars as pl
import xgboost as xgb
import torch
from torch import optim, nn
from rich import print
from rich.console import Console
import plotly.express as px

from sklearn.datasets import make_swiss_roll
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

RANDOM_SEED = 1


#loaded
samples = 300
whole = make_swiss_roll(n_samples = samples, random_state = RANDOM_SEED)[0]
data = whole[:, 0:2].copy()
label = whole[:, 2].copy()


# - models -

#decision-tree
dtree = DecisionTreeRegressor(
    criterion = 'squared_error', #default
    max_leaf_nodes = 100+1,
    random_state = RANDOM_SEED,
    )

#random-forest
rforest = RandomForestRegressor(
    criterion = 'squared_error', #default
    max_leaf_nodes = 100+1,
    verbose = 1,
    random_state = RANDOM_SEED,
    )

#gradient-boosting
booster = GradientBoostingRegressor(

    loss = 'squared_error', #default
    max_leaf_nodes = 10+1, max_depth = None,

    n_iter_no_change = None, #default
    validation_fraction = 0.03,

    learning_rate = 0.1, #default
    n_estimators = 300,
    verbose = 1,

    random_state = RANDOM_SEED,
    criterion = 'squared_error',

    )

#mlp
mlp = MLPRegressor(
    hidden_layer_sizes = [128, 64, 32, 16, 8, 4, 2],
    batch_size = 32,
    max_iter = 300,
    verbose = True,
    random_state = RANDOM_SEED,
    )


# - trained -

X = data.copy()
y = label.copy()

dtree.fit(X, y)
rforest.fit(X, y)
booster.fit(X, y)
mlp.fit(X, y)


# - plots -

fig1 = pp.figure(layout = 'constrained', facecolor = 'white')
gs1 = fig1.add_gridspec(nrows = 2, ncols = 1, height_ratios = [3, 4])
sgs1 = gs1[2-1].subgridspec(nrows = 1, ncols = 4)
fig1.suptitle('Swiss Roll Reconstruction')
ax1_1 = fig1.add_subplot(gs1[1-1], projection = '3d')
ax1_1.set(
    title = f"#samples: {samples}",
    box_aspect = [1, 1, 1],
    )
ax1_1.set_xlabel(r'$x_{1}$', size = 'large')
ax1_1.set_ylabel(r'$x_{2}$', size = 'large')
ax1_1.set_zlabel(r'$y$', size = 'medium')
ax1_1.tick_params(labelsize = 'x-small')
ax1_1.view_init(azim = -80, elev = 10)
ax1_2 = fig1.add_subplot(sgs1[1-1], projection = '3d', title = 'Decision Tree')
ax1_3 = fig1.add_subplot(sgs1[2-1], projection = '3d', title = 'Random Forest')
ax1_4 = fig1.add_subplot(sgs1[3-1], projection = '3d', title = 'Gradient Boosting')
ax1_5 = fig1.add_subplot(sgs1[4-1], projection = '3d', title = 'MLP')
for i in [ax1_2, ax1_3, ax1_4, ax1_5]:
    i.set(
        box_aspect = [1, 1, 1],
        xticklabels = [], yticklabels = [], zticklabels = [],
        )
    i.view_init(azim = -80, elev = 10)

label_ = ax1_1.plot(
    data[:, 0], data[:, 1], label,
    marker = 'o', markersize = 5, alpha = 0.5, color = 'grey',
    linestyle = '',
    label = 'label',
    )
pred_dtree_ = ax1_2.plot(
    data[:, 0], data[:, 1], dtree.predict(data),
    marker = 'o', markersize = 3, alpha = 0.5, color = 'tab:brown',
    linestyle = '',
    label = f"\
reconstructed (decision-tree)\n\
RMSE: {root_mean_squared_error(label, dtree.predict(data)):.3g}\
",
    ) #decision-tree
pred_rforest = ax1_3.plot(
    data[:, 0], data[:, 1], rforest.predict(data),
    marker = 'o', markersize = 3, alpha = 0.5, color = 'tab:red',
    linestyle = '',
    label = f"\
reconstructed (random-forest)\n\
RMSE: {root_mean_squared_error(label, rforest.predict(data)):.3g}\
",
    ) #random-forest
pred_booster_ = ax1_4.plot(
    data[:, 0], data[:, 1], booster.predict(data),
    marker = 'o', markersize = 3, alpha = 0.5, color = 'tab:green',
    linestyle = '',
    label = f"\
reconstructed (gradient-boosting)\n\
RMSE: {root_mean_squared_error(label, booster.predict(data)):.3g}\
",
    ) #gradient-boosting
pred_mlp_ = ax1_5.plot(
    data[:, 0], data[:, 1], mlp.predict(data),
    marker = 'o', markersize = 3, alpha = 0.5, color = 'tab:blue',
    linestyle = '',
    label = f"\
reconstructed (MLP)\n\
RMSE: {root_mean_squared_error(label, mlp.predict(data)):.3g}\
",
    ) #MLP
fig1.legend(loc = 'upper right', markerscale = 2)


#saved
os.makedirs('../figures', exist_ok = True)
fig1.savefig('../figures/swiss_roll_reconstruction.png', dpi = 300)
