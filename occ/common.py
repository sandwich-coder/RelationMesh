from copy import deepcopy as copy
import types, inspect
import multiprocessing as mp
import time
import pathlib
import numpy as np
import pyarrow as pa
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams.update({

    'figure.figsize': [10, 10],
    'figure.edgecolor': 'none',

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': 'none',

    'lines.markersize': 1,
    'lines.linewidth': 0.5,
    'scatter.edgecolors': 'none',

    'legend.facecolor': 'none',
    'legend.edgecolor': 'none',

    })

import polars as pl
import xgboost as xgb
import torch
from torch import nn, optim
from rich.console import Console
