import os, sys
sys.path.append('..')

from common import *
logger = logging.getLogger(name = 'swiss-hole')
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler

from models import (
    RelationMesh,
    Autoencoder,
    IsolationForest,
    )

#random
RANDOM_SEED = 42


#loaded
roll, main_axis = make_swiss_roll(n_samples = 1000, random_state = RANDOM_SEED)
rmesh = RelationMesh()
aencoder = Autoencoder()
iforest = IsolationForest()


# - prepared -

hole_idx0 = (roll[:, 0] < -1) & (roll[:, 1] > 6.) & (roll[:, 1] < 14.)
rest_idx0 = ~hole_idx0

hole_idx0 = np.where(hole_idx0)[0]
rest_idx0 = np.where(rest_idx0)[0]

roll_hole = roll[hole_idx0, :]
roll_rest = roll[rest_idx0, :]


# - trained -

X = pl.DataFrame(roll_rest, schema = ['x1', 'x2', 'x3'])

rmesh.fit(
    X,
    config = {
        'Scaler': StandardScaler,
        'regressor': {
            'max_delta_step': 0,
            'objective': 'reg:squarederror',
            'reg_lambda': 0,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 0,
            'device': 'cpu',
            'n_jobs': 1,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'max_bin': len(X),
            },
        },
    random_seed = RANDOM_SEED,
    )
aencoder.fit(X, hidden_layers = [128, 64, 32, 16, 8, 4, 2], dropout = 0, verbose = 2, random_seed = RANDOM_SEED)
iforest.fit(X, random_seed = RANDOM_SEED)


# - predicted -

data_t = pl.DataFrame(roll, schema = ['x1', 'x2', 'x3'])
label_t = np.zeros([len(data_t)], dtype = 'bool')
label_t[hole_idx0] = True

scores_t = {
    f"{rmesh}": rmesh.predict(data_t, return_scores = True)[1],
    f"{aencoder}": aencoder.predict(data_t, return_scores = True)[1],
    f"{iforest}": iforest.predict(data_t, return_scores = True)[1],
    }
for i in scores_t.keys(): #normalized
    score = scores_t[i]
    score = score / (score.max(axis = 0) - score.min(axis = 0))
    score = score - score.min(axis = 0)
    scores_t[i] = score


# - plot -

fig = pp.figure(layout = 'constrained', figsize = (10, 4.5))
gs = fig.add_gridspec(nrows = 2, ncols = 4)
ax_1 = fig.add_subplot(gs[1-1], projection = '3d')
ax_1.set(
    box_aspect = [1, 1, 1],
    title = 'Ground Truth',
    xticks = [],
    yticks = [],
    zticks = [],
    )
ax_1.view_init(azim = -70, elev = 10)
ax_2 = fig.add_subplot(gs[2-1], projection = '3d')
ax_2.set(
    box_aspect = [1, 1, 1],
    title = 'IsolationForest',
    xticks = [],
    yticks = [],
    zticks = [],
    )
ax_2.view_init(azim = -70, elev = 10)
ax_3 = fig.add_subplot(gs[3-1], projection = '3d')
ax_3.set(
    box_aspect = [1, 1, 1],
    title = 'Autoencoder',
    xticks = [],
    yticks = [],
    zticks = [],
    )
ax_3.view_init(azim = -70, elev = 10)
ax_4 = fig.add_subplot(gs[4-1], projection = '3d')
ax_4.set(
    box_aspect = [1, 1, 1],
    title = 'RelationMesh',
    xticks = [],
    yticks = [],
    zticks = [],
    )
ax_4.view_init(azim = -70, elev = 10)
ax_5 = fig.add_subplot(gs[5-1])
ax_5.set(
    box_aspect = 1,
    xticks = [],
    yticks = [],
    )
ax_5.spines['bottom'].set(visible = False)
ax_5.spines['left'].set(visible = False)
ax_6 = fig.add_subplot(gs[6-1])
ax_6.set(
    box_aspect = 1,
    xticks = [],
    yticks = [],
    )
ax_6.spines['bottom'].set(visible = False)
ax_6.spines['left'].set(visible = False)
ax_7 = fig.add_subplot(gs[7-1])
ax_7.set(
    box_aspect = 1,
    xticks = [],
    yticks = [],
    )
ax_7.spines['bottom'].set(visible = False)
ax_7.spines['left'].set(visible = False)
ax_8 = fig.add_subplot(gs[8-1])
ax_8.set(
    box_aspect = 1,
    xticks = [],
    yticks = [],
    )
ax_8.spines['bottom'].set(visible = False)
ax_8.spines['left'].set(visible = False)

cmap = 'OrRd'
plot_1 = ax_1.scatter(
    *data_t.to_numpy(writable = False).transpose(1, 0),
    s = 11, c = label_t.astype('float64'), cmap = cmap, alpha = 0.8,
    )
plot_2 = ax_2.scatter(
    *data_t.to_numpy(writable = False).transpose(1, 0),
    s = 11, c = scores_t['isolation-forest'], cmap = cmap, alpha = 0.8,
    )
plot_3 = ax_3.scatter(
    *data_t.to_numpy(writable = False).transpose(1, 0),
    s = 11, c = scores_t['autoencoder'], cmap = cmap, alpha = 0.8,
    )
plot_4 = ax_4.scatter(
    *data_t.to_numpy(writable = False).transpose(1, 0),
    s = 11, c = scores_t['relation-mesh'], cmap = cmap, alpha = 0.8,
    )
plot_5 = ax_5.scatter(
    main_axis, data_t[:, 'x2'].to_numpy(writable = False),
    c = label_t.astype('float64'),
    cmap = cmap,
    s = 17,
    )
plot_6 = ax_6.scatter(
    main_axis, data_t[:, 'x2'].to_numpy(writable = False),
    c = scores_t['isolation-forest'],
    cmap = cmap,
    s = 17,
    )
plot_7 = ax_7.scatter(
    main_axis, data_t[:, 'x2'].to_numpy(writable = False),
    c = scores_t['autoencoder'],
    cmap = cmap,
    s = 17,
    )
plot_8 = ax_8.scatter(
    main_axis, data_t[:, 'x2'].to_numpy(writable = False),
    c = scores_t['relation-mesh'],
    cmap = cmap,
    s = 17,
    )
cbar = fig.colorbar(plot_8, ax = ax_8, shrink = 0.95, label = 'anomaly score')


#saved
os.makedirs('../../figures', exist_ok = True)
fig.savefig('../../figures/swiss-hole.png', dpi = 300)
