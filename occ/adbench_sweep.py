if __name__ != '__main__':
    raise RuntimeError('This module is a standalone experiment.')
from rich.traceback import install as rich_traceback
rich_traceback()

from common import *
import logging
logger = logging.getLogger()
logging.basicConfig(level = 'INFO')
from datetime import datetime as Datetime
from sklearn.preprocessing import StandardScaler

from loader import Loader
from models import (
    RelationMesh,
    IsolationForest,
    OCSVM,
    Autoencoder,
    ECOD,
    )
from evaluate import fast_result, aucplot

#constants
RANDOM_SEEDS = list(range(1, 11))
MAX_P_ANOMALOUS = None
TRAIN_FPR = 0.01
RESULT_DIR = pathlib.Path('temp/adbenchsweep')

logger.info(f"Random Seeds: {RANDOM_SEEDS}")
logger.info(f"Maximum Anomaly Ratio: {MAX_P_ANOMALOUS}")
logger.info(f"Train fpr: {TRAIN_FPR}")
logger.info(f"Result Directory: {RESULT_DIR}")

datalist = [
    'adbench/1_ALOI',
    'adbench/2_annthyroid',
    'adbench/3_backdoor',
    'adbench/4_breastw',
    'adbench/5_campaign',
    'adbench/6_cardio',
    'adbench/7_Cardiotocography',
    'adbench/8_celeba',
    'adbench/9_census',
    'adbench/10_cover',
    'adbench/11_donors',
    'adbench/12_fault',
    'adbench/13_fraud',
    'adbench/14_glass',
    'adbench/15_Hepatitis',
    'adbench/16_http',
    'adbench/17_InternetAds',
    'adbench/18_Ionosphere',
    'adbench/19_landsat',
    'adbench/20_letter',
    'adbench/21_Lymphography',
    'adbench/22_magic.gamma',
    'adbench/23_mammography',
    'adbench/24_mnist',
    'adbench/25_musk',
    'adbench/26_optdigits',
    'adbench/27_PageBlocks',
    'adbench/28_pendigits',
    'adbench/29_Pima',
    'adbench/30_satellite',
    'adbench/31_satimage-2',
    'adbench/32_shuttle',
    'adbench/33_skin',
    'adbench/34_smtp',
    'adbench/35_SpamBase',
    'adbench/36_speech',
    'adbench/37_Stamps',
    'adbench/38_thyroid',
    'adbench/39_vertebral',
    'adbench/40_vowels',
    'adbench/41_Waveform',
    'adbench/42_WBC',
    'adbench/43_WDBC',
    'adbench/44_Wilt',
    'adbench/45_wine',
    'adbench/46_WPBC',
    'adbench/47_yeast',
    ]


#selected
datasets = []
for i in datalist:

    loader = Loader()
    normal_t, normal_v = loader.load(i, split_seed = 1)

    if len(normal_t.columns) < 20 or len(normal_t.columns) > 99 or len(normal_t) + len(normal_v) < 1000:
        pass
    else:
        print(f"dataset: {i} | #samples: {len(normal_t) + len(normal_v)} | #features: {len(normal_t.columns)}")
        datasets.append(i)

aucrocs = {}
aucprs = {}
f1s = {}
for i in RANDOM_SEEDS:
    logger.info(f"random seed: {i}")

    aucrocs[i] = []
    aucprs[i] = []
    f1s[i] = []
    for ii in datasets:
        logger.info(f"dataset: {ii}")


        # - loaded -

        loader = Loader()
        normal_t, normal_v = loader.load(ii, split_seed = i)

        #models
        models = [
            RelationMesh(base_learner = 'RF'),
            IsolationForest(),
            OCSVM(),
            Autoencoder(),
            ECOD(),
            ]


        # - trained -

        #training set
        X = normal_t.clone()

        #train
        for iii in models:
            if f"{iii}" == 'relation-mesh': #my model
                iii.fit(
                    X,
                    categorical = loader.get_catcols(ii),
                    config = config.get(ii),
                    random_seed = i,
                    )
            else:
                iii.fit(X, categorical = loader.get_catcols(ii), random_seed = i)


        # - evaluation -

        #prepared
        if MAX_P_ANOMALOUS is not None:
            anomalous_t, anomalous_v = loader.load(
                ii,
                anomalous = True,
                samples = int((len(normal_t) + len(normal_v)) * (MAX_P_ANOMALOUS / (1 - MAX_P_ANOMALOUS))),
                split_seed = i,
                sample_seed = i,
                )
        else:
            anomalous_t, anomalous_v = loader.load(
                ii,
                anomalous = True,
                split_seed = i,
                )

        #combined
        label_t = np.array([False] * len(normal_t) + [True] * len(anomalous_t), dtype = 'bool')
        data_t = pl.concat([normal_t, anomalous_t], how = 'vertical')
        label_v = np.array([False] * len(normal_v) + [True] * len(anomalous_v), dtype = 'bool')
        data_v = pl.concat([normal_v, anomalous_v], how = 'vertical')

        #scores
        predictions = {}
        ranks = {}
        for iii in models:
            predictions[f"{iii}"], ranks[f"{iii}"] = iii.predict(data_v, train_fpr = TRAIN_FPR, return_ranks = True)

        scores = fast_result(label_v, predictions, return_values = True)
        aucs = aucplot(label_v, ranks, return_values = True)

        aucrocs_bymodel = {}
        aucprs_bymodel = {}
        f1s_bymodel = {}
        for iii in predictions.keys():
            aucrocs_bymodel[iii] = aucs[iii]['roc']
            aucprs_bymodel[iii] = aucs[iii]['pr']
            f1s_bymodel[iii] = scores[iii]['F1']
        aucrocs[i].append(aucrocs_bymodel)
        aucprs[i].append(aucprs_bymodel)
        f1s[i].append(f1s_bymodel)



#to frame and averaged
for i in [aucrocs, aucprs, f1s]:

    temp = [] #mean
    for ii in i.keys():
        i[ii] = pl.DataFrame(i[ii])

        temp.append(i[ii].to_numpy(writable = True))

    i['mean'] = pl.DataFrame(
        np.stack(temp, axis = 0).mean(axis = 0, dtype = 'float64'),
        schema = [f"{c}" for c in models],
        )


#dataset column added
for i in [aucrocs, aucprs, f1s]:
    for ii in i.keys():
        i[ii] = i[ii].with_columns(
            dataset = pl.Series(datasets),
            )
        temp = i[ii].columns
        i[ii] = i[ii].select(temp[-1:] + temp[:-1])

#make table
for i in [aucrocs, aucprs, f1s]:
    table = i['mean']

    columns = table.columns[1:]
    index_col = table.columns[0]

    indices = table[:, index_col].to_list()
    indices.append('Average')

    values = table.drop([index_col]).to_numpy(writable = True)
    values = np.concatenate([
        values,
        values.mean(keepdims = True, axis = 0, dtype = 'float64'),
        ], axis = 0).round(decimals = 3)

    table = pl.DataFrame(values, schema = columns).with_columns(**{
        index_col: pl.Series(indices),
        })
    table = table.select(table.columns[-1:] + table.columns[:-1])
    i['table'] = table



#saved
timestamp = Datetime.now().strftime('%y%m%d_%H%M%S')
result_dir = pathlib.Path(f"{RESULT_DIR}-{timestamp}")
result_dir.mkdir(parents = True, exist_ok = True)
for i in aucrocs.keys():
    aucrocs[i].write_csv(f"{result_dir}/aucroc-{i}.csv")
    aucprs[i].write_csv(f"{result_dir}/aucpr-{i}.csv")
    f1s[i].write_csv(f"{result_dir}/f1-{i}.csv")
