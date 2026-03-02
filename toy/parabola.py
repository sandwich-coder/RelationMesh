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

    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.facecolor':'none',

    'lines.markersize':1,
    'lines.linewidth':0.5,

    'legend.facecolor':'none',
    'legend.edgecolor':'none',

    })

import polars as pl
import torch
from torch import optim, nn
from rich import print
from rich.console import Console
import plotly.express as px

from numpy.random import default_rng as generate_rng


x1 = np.linspace(0, 3, axis = 0, dtype = 'float64', endpoint = False, num = 100000)
x2_up = np.sqrt(x1, dtype = 'float64')
x2_down = -np.sqrt(x1, dtype = 'float64')

x1_up_actual = np.array([
    0.1, 0.4,
    2.2, 2.5, 2.6,
    ], dtype = 'float64')
x1_down_actual = np.array([
    1.1, 1.2, 1.6, 1.9,
    ], dtype = 'float64')
x2_up_actual = np.sqrt(x1_up_actual, dtype = 'float64')
x2_down_actual = -np.sqrt(x1_down_actual, dtype = 'float64')
x1_actual = np.concatenate([x1_up_actual, x1_down_actual], axis = 0)
x2_actual = np.concatenate([x2_up_actual, x2_down_actual], axis = 0)

idx_hop1 = np.where(x1 < 1)[0]
idx_hop2 = np.where((x1 >= 1) & (x1 < 2))[0]
idx_hop3 = np.where(x1 >= 2)[0]
x2_path = np.concatenate([
    x2_up[idx_hop1],
    x2_down[idx_hop2],
    x2_up[idx_hop3],
    ], axis = 0)

fig1 = pp.figure(layout = 'constrained')
ax1 = fig1.add_subplot()
ax1.set(
    box_aspect = 1,
    xlabel = 'x',
    ylabel = 'y',
    )
ax1.set_xlabel(ax1.get_xlabel(), size = 'x-large')
ax1.set_ylabel(ax1.get_ylabel(), size = 'x-large')
pp.setp(ax1.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

plot1_1, = ax1.plot(
    x1, x2_up,
    marker = '', color = 'slategrey',
    linestyle = '-', linewidth = 3,
    label = 'branch1',
    )
plot1_2, = ax1.plot(
    x1, x2_down,
    marker = '', color = 'slategrey',
    linestyle = '--', linewidth = 3,
    label = 'branch2',
    )
ax1.text(2, 1.3, r'$y=\sqrt{x}$', ha = 'left', va = 'center', fontsize = 30)
ax1.text(2, -1.3, r'$y=-\sqrt{x}$', ha = 'left', va = 'center', fontsize = 30)
ax1.legend(loc = 'right', fontsize = 'xx-large')

fig2 = pp.figure(layout = 'constrained', figsize = (5, 4))
ax2 = fig2.add_subplot()
ax2.set(
    box_aspect = 0.75,
    xticks = [],
    yticks = [],
    )
ax2.spines['bottom'].set(visible = False)
ax2.spines['left'].set(visible = False)

plot2_1, = ax2.plot(
    x1, x2_up,
    marker = '', color = 'slategrey',
    linestyle = '-', linewidth = 0.8,
    label = 'branch 1',
    )
plot2_2, = ax2.plot(
    x1, x2_down,
    marker = '', color = 'slategrey',
    linestyle = '--', linewidth = 0.8,
    label = 'branch 2',
    )
plot2_3 = ax2.scatter(
    x1_actual, x2_actual,
    s = 20, c = 'tab:blue',
    label = '\nData\n',
    )
plot2_4, = ax2.plot(
    x1, x2_path,
    marker = '', color = 'tab:blue',
    linestyle = '-', linewidth = 1.2,
    label = 'Function\nApproximated',
    )
ax2.legend(
    handles = [
        plot2_3,
        plot2_1,
        plot2_2,
        plot2_4,
        ],
    loc = 'lower left', fontsize = 'medium',
    )

os.makedirs('../figures', exist_ok = True)
fig1.savefig('../figures/parabola.png', dpi = 300)
fig2.savefig('../figures/piecewise-approximation.png', dpi = 300)
