from common import *
import logging
logger = logging.getLogger(name = __name__)
from datetime import datetime as Datetime
from scipy import integrate
from sklearn.metrics import (

    precision_score,
    recall_score,
    f1_score,

    classification_report,
    precision_recall_curve,
    roc_curve,

    )

time_start = Datetime.now().strftime('%y%m%d_%H%M%S')
logger.info(f"Timestamp: {time_start}")


def fast_result(label, predictions, return_values = False):
    if not isinstance(label, np.ndarray):
        raise TypeError(f"The label should be a numpy.ndarray. Received {type(label)}.")
    if not isinstance(predictions, dict):
        raise TypeError(f"The predictions should be a dictionary. Received {type(predictions)}.")
    if not isinstance(return_values, bool):
        raise TypeError('\'return_values\' should be boolean.')

    for i in predictions.keys():
        print(f"{i} >>")
        print(classification_report(label, predictions[i]))

    if return_values:
        results = {}
        for i in predictions.keys():
            results[i] = {
                'precision': precision_score(label, predictions[i]),
                'recall': recall_score(label, predictions[i]),
                'F1': f1_score(label, predictions[i]),
                }

        return results



def aucplot(label, ranks, save_plots = False, prefix = '', timestamp = False, return_values = False):
    if not isinstance(label, np.ndarray):
        raise TypeError(f'The label should be a numpy.ndarray. Received {type(label)}.')
    if not isinstance(ranks, dict):
        raise TypeError(f'The ranking should be a dictionary. Received {type(label)}.')
    if not isinstance(save_plots, bool):
        raise TypeError('\'save_plots\' should be boolean.')
    if not isinstance(prefix, str):
        raise TypeError('\'prefix\' should be a string.')
    if not isinstance(timestamp, bool):
        raise TypeError('\'timestamp\' should be boolean.')
    if not isinstance(return_values, bool):
        raise TypeError('\'return_values\' should be boolean.')

    aucs = {}
    for i in ranks.keys():
        aucs[i] = {}


    # - auc-pr -

    fig1 = pp.figure(layout = 'constrained')
    fig1.suptitle('AUC-PR')
    ax1 = fig1.add_subplot(
        box_aspect = 1,
        xlabel = 'recall',
        ylabel = 'precision',
        )
    ax1.tick_params(axis = 'y', labelrotation = 90)
    pp.setp(ax1.get_yticklabels(), va = 'center')

    for i in ranks.keys():
        precision, recall, temp = precision_recall_curve(label, ranks[i])
        auc = -integrate.trapezoid(precision, x = recall, axis = 0).tolist()
        aucs[i]['pr'] = auc

        plot1, = ax1.plot(
            recall, precision,
            marker = '', alpha = 0.6,
            linestyle = '-', linewidth = 3,
            label = f"{i}: {auc:.3f}",
            )

    ax1.legend()


    # - auc-roc -

    fig2 = pp.figure(layout = 'constrained')
    fig2.suptitle('AUC-ROC')
    ax2 = fig2.add_subplot(
        box_aspect = 1,
        xlabel = 'false-positive rate',
        ylabel = 'true-positive rate',
        )
    ax2.tick_params(axis = 'y', labelrotation = 90)
    pp.setp(ax2.get_yticklabels(), va = 'center')

    for i in ranks.keys():
        fpr, tpr, temp = roc_curve(label, ranks[i])
        auc = integrate.trapezoid(tpr, x = fpr, axis = 0).tolist()
        aucs[i]['roc'] = auc

        plot2, = ax2.plot(
            fpr, tpr,
            marker = '', alpha = 0.6,
            linestyle = '-', linewidth = 3,
            label = f"{i}: {auc:.3f}",
            )

    ax2.legend()


    if save_plots:

        prefix = prefix.replace('/', '-')
        if timestamp:
            ts = time_start
            ts = '-' + ts
        else:
            ts = ''
        fig_dir = pathlib.Path(f"figures/{prefix}aucs{ts}")

        fig_dir.mkdir(parents = True, exist_ok = True)
        fig1.savefig(f"{fig_dir}/aucpr.png", dpi = 300)
        fig2.savefig(f"{fig_dir}/aucroc.png", dpi = 300)


    if return_values:
        return aucs


def scoreplot(label, scores, save_plots = False, prefix = '', timestamp = False):
    if not isinstance(label, np.ndarray):
        raise TypeError(f"The label should be a numpy.ndarray. Received {type(label)}")
    if not isinstance(scores, dict):
        raise TypeError(f"The scores should be a dictionary. Received {type(scores)}")
    if not isinstance(save_plots, bool):
        raise TypeError('\'save_plots\' should be boolean.')
    if not isinstance(prefix, str):
        raise TypeError('\'prefix\' should be a string.')
    if not isinstance(timestamp, bool):
        raise TypeError('\'timestamp\' should be boolean.')

    figs = {}
    for i in scores.keys():
        fig = pp.figure(layout = 'constrained', figsize = (20, 7))
        ax = fig.add_subplot()
        ax.set(
            box_aspect = 7 / 20,
            title = f"{i}",
            xlabel = 'score',
            ylabel = 'proportion (%)',
            )
        ax.tick_params(axis = 'y', labelrotation = 90)
        pp.setp(ax.get_yticklabels(), va = 'center')

        score_normal = scores[i][~label]
        score_anomalous = scores[i][label]
        plot = ax.hist(
            [score_normal, score_anomalous],
            bins = 300,
            weights = [

                #separately normalized
                np.ones(score_normal.shape, dtype = 'float64') / float(len(score_normal)) * 100.,
                np.ones(score_anomalous.shape, dtype = 'float64') / float(len(score_anomalous)) * 100.,

                ],
            range = np.quantile(scores[i], [0, 0.999], axis = 0).tolist(),
            color = ['tab:blue', 'tab:red'],
            align = 'left',
            rwidth = 0.9,
            label = ['Normal', 'Anomalous'],
            )

        ax.legend()
        figs[i] = fig


    if save_plots:

        prefix = prefix.replace('/', '-')
        if timestamp:
            ts = time_start
            ts = '-' + ts
        else:
            ts = ''
        fig_dir = pathlib.Path(f"figures/{prefix}scores{ts}")

        fig_dir.mkdir(parents = True, exist_ok = True)
        for i in figs.keys():
            figs[i].savefig(f"{fig_dir}/{i}.png", dpi = 600)
