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
    'figure.facecolor':'white',
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.facecolor':'none',
    'lines.markersize':1,
    'lines.linewidth':0.5,
    'legend.edgecolor':'none',
    'legend.facecolor':'none',
    })

import polars as pl
import torch
from torch import optim, nn
from rich import print
from rich.console import Console

from numpy.random import default_rng
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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
label = perturb(label, width = 0.3) #perturbed
label[3000:3100] = 0.5 #noise

#train-val split
idx_val = np.zeros(len(data), dtype = 'bool')
idx_val[5000:6000] = True
data_t = data[~idx_val]
label_t = label[~idx_val]
data_v = data[idx_val]
label_v = data[idx_val]


#y limit of the first graph
ylim = label - label_t.mean(axis = 0, dtype = 'float64', keepdims = True)
ylim = ylim * 1.1
ylim = [ylim.min(axis = 0).tolist(), ylim.max(axis = 0).tolist()]



" ========== Gradient Boosting ========== "
""" -------------- NOTE --------------

  The gradient boosting is said a gradient descent in the function space.
However, the available "directions" of each descent is limited by the capacity of the individual functions added,
like the situation in which we can choose only east-west-south-north movement in a 2-dimensional space,
unless the 'max_leaves' is larger than or equal to the training size, the directions becoming virtually ideal.
As a consequence, every step will deviate from the "true direction" of gradient
and the smaller the step size the closer the actual path resembles the ideal one.
The phenomenon of the early fast learning and late slow convergence induced by an insufficient capacity and a large learning rate
is often a result of the descent path having deviated from the ideal.
With a small learning rate the wrong directions of shallow trees follow the ideal path in high-resolution-approximation,
but take more steps since the actual distance needed to reach the same optimum
tends to be proportional to the cosines of the projection angles.

  Until now, it was about fitting to the training set PERFECTLY.
In most cases however, there are labels we do NOT want to fit, which are called noise.
Why noise should not be fit to is not about generalization.
They are such labels the model should be wrong
even when given as an input the exact same instance in the training set.
One major way not to learn noise is the restriction of capacity.
  Restriction of capacity refers to restricting the individual trees shallow.
Given the learning rate is small enough shallow trees approximate the ideal path,
but differ in the iteration where the splits start to realize rare spikes or prevalent perturbations.
This strategy works for any loss-functions, but effective with such outlier-robust losses as absolute-error.

  Most important, the model must be able to generalize to unseen inputs. One way to do is restricting sight.
Resctiction of sight refers to restricting the training set each tree fits to
by sampling the training set for every new tree built.
Sampling corresponds to minibatch gradient descent, a stochastic gradient descent,
hence a good approximation of the ideal path.
When each tree sees a subset, the points become sparser
and the "constant region" of the prediction for an input continues to where other inputs that had not been sampled would reside,
"diluting" the next prediction with different samples and making every point influential to the nearby.
This "dilution" is the core aspect of sampling.
  Yet, an interesting property of sampling is the emergence of new splits.
For example, possible split points for input (1, 2, 3, 8, 9, 10) is 1.5, 2.5, 5.5, 8.5, and 9.5.
If (2, 8) is sampled new split 5, and if (3, 9) is sampled new split 6 emerges.
This phenomenon is key to split-smoothing and SOFTENS the interpolation from "stepwise" to "linear".
Note the 'split-smoothing' is not the main property. Decision-tree is piecewise-constant by definition.
On one hand, Lowering the sampling rate too low, obsessed at generalization, may allow spikes to "lift" the nearby to disrupt the model.
On the other hand, if the goal is not "hit the answer" but "follow the trend", small rate may be the start.

In the original gradient-boosting, every tree minimizes the variance by setting a leaf as mean of the targets, or negative derivative of the loss function. In other words, its leaf is determined as mean of the gradient steps. If we tweak this a bit, it becomes second-slope-weighted average of the newton steps, the trick introduced by the xgboost. It is called 'newton-boosting'.

"""


# It seems the subsample being 10% produces the linear interpolation
# as wide as roughly 20 times the mean interval of trainng set.
# rough estimate:
#     80%: 1x
#     50%: 3x
#     30%: 5x
#     10%: 20x
# In the newton-boostings however, subsampling does not introduce new splits.
#config
loss_function = 'squared_error'
subsample = 0.1
tree_config = {

    'max_leaf_nodes': 3, 'max_depth': None,

    'min_samples_leaf': 1,

    # Whatever the boosting loss may be, the indiviual trees minimize the variance with the squared-error.

    }
learning_rate = 0.01
iterations = 3000

X = data_t.copy()
y = label_t.copy() #initial target
if loss_function == 'squared_error':
    prediction_t = y.mean(axis = 0, dtype = 'float64', keepdims = True) #initial train-output
elif loss_function == 'absolute_error':
    prediction_t = np.median(y, axis = 0, keepdims = True) #initial train-output
prediction_v = prediction_t.copy()

snap = 30
assert snap <= iterations, 'The number of snapshots greater than the iterations.'
snap = np.linspace(1, iterations, axis = 0, dtype = 'int64', endpoint = True, num = snap).tolist()
for i in range(1, iterations+1):

    # The target of each tree is not the error of the total prediction.
    # Each new target is the negative derivative of the loss function
    # with respect to the total prediction.
    # The new target being the same as the error itself in the squared-error loss is just a special case.
    # Each tree then fits to the new target by minimizing the variance.
    # A leaf value is determined as mean of the negative slopes, mean of the gradient steps.
    #new target
    if loss_function == 'squared_error':
        y = label_t - prediction_t
    elif loss_function == 'absolute_error':
        y = np.sign(label_t - prediction_t, dtype = 'float64')
    X = data_t.copy()

    #sampled
    idx_sample = rng.choice(
        np.arange(len(y)),
        axis = 0,
        replace = False,
        shuffle = False,
        size = int(len(y) * subsample),
        )
    idx_sample.sort(axis = 0)
    X = X[idx_sample]
    y = y[idx_sample]

    model = DecisionTreeRegressor(**tree_config)
    model.fit(X, y)
    pred = model.predict(X)

    pred_t = model.predict(data_t)
    prediction_t = prediction_t + learning_rate * pred_t

    #follows the same process for the validation
    # See how it diverges from the training
    pred_v = model.predict(data_v)
    prediction_v = prediction_v + learning_rate * pred_v


    # - plots -
    if i in snap:

        fig = pp.figure(layout = 'constrained', facecolor = 'white')
        gs = fig.add_gridspec(nrows = 2, ncols = 1)
        fig.suptitle(f'Curve Fit <iteration {i}>')
        ax_1 = fig.add_subplot(gs[1-1])
        ax_1.set(
            box_aspect = 0.5,
            ylim = ylim,
            xticks = [], yticks = [],
            )
        ax_1.spines['left'].set(visible = False)
        ax_1.axhline(0, linestyle = '-.', linewidth = 2, color = 'brown')
        ax_2 = fig.add_subplot(gs[2-1])
        ax_2.set(
            box_aspect = 0.5,
            aspect = 1,
            )
        pp.setp(ax_2.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')
        if loss_function == 'squared_error':
            temp = label.mean(axis = 0, dtype = 'float64')
        elif loss_function == 'absolute_error':
            temp = np.median(label, axis = 0)
        ax_2.axhline(temp, linestyle = '-.', linewidth = 2, color = 'brown', label = 'center')

        plot_1_1, = ax_1.plot(
            X[:, 0], y,
            marker = 'x', markersize = 7, color = 'grey',
            linestyle = '',
            )
        plot_1_2, = ax_1.plot(
            X[:, 0], pred,
            marker = '', color = 'blue',
            linestyle = '--',
            )
        plot_2_1, = ax_2.plot(
            data[:, 0], label,
            marker = 'o', markersize = 0.1, color = 'black',
            linestyle = '',
            label = 'label',
            )
        plot_2_2, = ax_2.plot(
            data_t[:, 0], prediction_t,
            marker = '+', markersize = 3, color = 'blue', alpha = 0.5,
            linestyle = '',
            label = 'prediction (train)',
            )
        plot_2_3, = ax_2.plot(
            data_v[:, 0], prediction_v,
            marker = '+', markersize = 3, color = 'green', alpha = 0.5,
            linestyle = '',
            label = 'prediction (val)',
            )
        ax_2.legend()







" ========== Bagging ========== "


'''
#config
subsample = 0.8
replace = False
tree_config = {

    'criterion': 'absolute_error',
    'min_samples_leaf': 30,

    'max_leaf_nodes': None, 'max_depth': None,

    }
iterations = 300

X_, y_, pred_ = [], [], []
prediction_t = []
prediction_v = []
snap = 30
assert snap <= iterations, 'The number of snapshots greater than the iterations.'
snap = np.linspace(1, iterations, axis = 0, dtype = 'int64', endpoint = True, num = snap).tolist()
for i in range(iterations):

    #new
    idx_sample = rng.choice(
        np.arange(len(label_t)),
        axis = 0,
        replace = replace,
        shuffle = False,
        size = int(len(label_t) * subsample),
        )
    idx_sample.sort(axis = 0)
    y = label_t[idx_sample]
    X = data_t[idx_sample]

    #trained
    model = DecisionTreeRegressor(**tree_config)
    model.fit(X, y)

    #bootstrapped instances stored
    X_.append(X)
    y_.append(y)
    pred_.append(model.predict(X))

    #train
    prediction_t.append(model.predict(data_t))

    #val
    prediction_v.append(model.predict(data_v))

X_ = np.stack(X_, axis = 0)
y_ = np.stack(y_, axis = 0)
pred_ = np.stack(pred_, axis = 0)

prediction_t = np.stack(prediction_t, axis = 0)
prediction_v = np.stack(prediction_v, axis = 0)

for i in range(1, iterations+1):

    if i in snap:

        fig = pp.figure(layout = 'constrained', facecolor = 'white')
        gs = fig.add_gridspec(nrows = 2, ncols = 1)
        fig.suptitle(f'Curve Fit <iteration {i}>')
        ax_1 = fig.add_subplot(gs[1-1])
        ax_1.set(
            box_aspect = 0.5,
            ylim = ylim,
            xticks = [], yticks = [],
            )
        ax_1.spines['left'].set(visible = False)
        ax_1.axhline(0, linestyle = '-.', linewidth = 2, color = 'brown')
        ax_2 = fig.add_subplot(gs[2-1])
        ax_2.set(
            box_aspect = 0.5,
            aspect = 1,
            )
        pp.setp(ax_2.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')
        ax_2.axhline(label.mean(axis = 0), linestyle = '-.', linewidth = 2, color = 'brown', label = 'center')

        plot_1_1, = ax_1.plot(
            X_[i-1][:, 0], y_[i-1],
            marker = 'x', markersize = 2, color = 'grey',
            linestyle = '',
            )
        plot_1_2, = ax_1.plot(
            X_[i-1][:, 0], pred_[i-1],
            marker = '', color = 'blue',
            linestyle = '--',
            )
        plot_2_1, = ax_2.plot(
            data[:, 0], label,
            marker = 'o', markersize = 0.1, color = 'black',
            linestyle = '',
            label = 'label',
            )
        plot_2_2, = ax_2.plot(
            data_t[:, 0], prediction_t[:i].mean(axis = 0, dtype = 'float64'),
            marker = '+', markersize = 3, color = 'blue', alpha = 0.5,
            linestyle = '',
            label = 'prediction (train)',
            )
        plot_2_3, = ax_2.plot(
            data_v[:, 0], prediction_v[:i].mean(axis = 0, dtype = 'float64'),
            marker = '+', markersize = 3, color = 'green', alpha = 0.5,
            linestyle = '',
            label = 'prediction (val)',
            )
        ax_2.legend()
'''
