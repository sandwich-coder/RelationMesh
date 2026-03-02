from common import *
import logging
logger = logging.getLogger(name = __name__)
from xgboost import (

    XGBRegressor,
    XGBClassifier,

    XGBRFRegressor,
    XGBRFClassifier,

    )
from scipy import stats
from sklearn.preprocessing import (

    #numerical target
    RobustScaler,
    StandardScaler, #random-forest

    #categorical target
    OrdinalEncoder,

    )
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import randint
from math import sqrt
import platform
from psutil import cpu_count
import tempfile

from ..processor import Processor
from .learners import (

    RandomForestRegressor,
    RandomForestClassifier,

    MLPRegressor,
    MLPClassifier,

    )

#device
if torch.cuda.is_available():
    device = 'cuda'
    logger.info('CUDA is available, GPU used.')
else:
    device = 'cpu'
    logger.info('CUDA is not available, falling back to CPU.')

#threads
cpus = cpu_count(logical = False) #physical
n_jobs = cpus - 1
if platform.system() == 'Linux':
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    if 'GenuineIntel' in cpuinfo: #server
        n_jobs = 4
elif platform.system() == 'Darwin': #local
    n_jobs = max(n_jobs - 4, 1)
logger.info(f"#threads: {n_jobs}")


class RelationMesh:
    def __init__(self, base_learner = 'default'):
        if not isinstance(base_learner, str):
            raise TypeError('\'base_learner\' should be a string.')
        self._processor = Processor()
        self._learner = base_learner

    def __repr__(self):
        if self._learner == 'RF':
            return 'rmesh-RF'
        elif self._learner == 'MLP':
            return 'rmesh-MLP'
        elif self._learner == 'TabPFN':
            return 'rmesh-TabPFN'
        elif self._learner == 'rf':
            return 'rmesh-rf'
        else:
            return 'relation-mesh'
    def get_name(self):
        return 'RelationMesh'

    def fit(
        self,

        param_train,
        categorical = None,
        other = '..other',
        config = None,
        p_eval = 0.03,

        random_seed = None,
        verbose = 10,

        ):
        if not isinstance(param_train, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(categorical, list|None):
            raise TypeError('\'categorical\' should be a list.')
        if not isinstance(other, str):
            raise TypeError('\'other\' should be a string.')
        if not isinstance(config, dict|None):
            raise TypeError('\'config\' should be a dictionary.')
        if not isinstance(p_eval, float|int):
            raise TypeError('\'p_eval\' should be a real.')
        if not isinstance(random_seed, int|None):
            raise TypeError('\'random_seed\' should be an integer.')
        if not isinstance(verbose, int):
            raise TypeError('\'verbose\' should be an integer.')
        assert 0 <= p_eval < 1, 'evaluation proportion not in [0, 1)'
        if random_seed is not None:
            assert random_seed >= 0, 'random seed not natural'
        assert verbose >= 0, 'verbose negative'
        if categorical is None:
            categorical = []
        if config is None:
            config = {}
        if random_seed is None:
            random_seed = randint(0, 255) # The xgboost default is not random.
        train = param_train.clone(); del param_train

        #processor fit
        self._processor.fit(train, categorical, other)

        #transformed
        train = self._processor.transform(train)
        logger.info(f"Training Size: {train.shape} | %eval: {100 * p_eval:.2g}%")


        # - train -

        self.ensemble = {}
        self.trainerrors = {}
        for i in train.columns:
            y = train[:, i].to_numpy(writable = True)
            if y.dtype == 'object':
                y = y.astype('str')
            X = train.drop([i])

            #trivial case
            if len(np.unique(y)) < 2:
                logger.info(f"The target \'{i}\' is singular, falling back to rule-based indicator.")

                self.trainerrors[i] = np.zeros(y.shape, dtype = 'float64')

                self.ensemble[i] = {
                    'type': 'indicator',
                    'model': None,
                    'value': y[0].tolist(),
                    }
                continue


            #pipe-target
            if i in categorical:
                if self._learner != 'default':
                    encoder = OrdinalEncoder(categories = [self._processor.categories[i].categories.to_list()], dtype = 'int64')
                else:

                    encoder = config.get('Encoder', OrdinalEncoder)(categories = [self._processor.categories[i].categories.to_list()], dtype = 'int64')

                encoder.fit(y.reshape([y.shape[0], -1]))

            else:
                if self._learner != 'default':
                    scaler = StandardScaler(copy = True)
                else:

                    scaler = config.get('Scaler', RobustScaler)(copy = True) # This works only because every instantiation differs in its address. If the dictionary were to contain an instance of 'Scaler' class, copy is necessary.

                scaler.fit(y.reshape([y.shape[0], -1]))


            if i in categorical:
                y = encoder.transform(y.reshape([y.shape[0], -1]))[:, 0]

                #different base-learners
                if self._learner == 'RF':
                    model = RandomForestClassifier(min_samples_leaf = min(1000, len(X)//100+1), random_seed = random_seed, n_jobs = n_jobs)
                    model.fit(
                        X, y,
                        category_columns = [c for c in self._processor.categories.keys() if c != i],
                        )
                elif self._learner == 'MLP':
                    model = MLPClassifier(random_seed = random_seed)
                    model.fit(
                        X, y,
                        category_columns = [c for c in self._processor.categories.keys() if c != i],
                        )
                elif self._learner == 'rf':
                    model = XGBRFClassifier(
                        min_child_weight = min(1000., len(X)/100) * 0.09623,
                        subsample = 0.5,
                        colsample_bytree = 1,
                        colsample_bylevel = 1,
                        colsample_bynode = sqrt(len(X.columns)) / float(len(X.columns)),
                        max_depth = 0,
                        enable_categorical = True,
                        objective = 'multi:softmax',
                        eval_metric = 'mlogloss',
                        tree_method = 'hist',
                        random_state = random_seed,
                        device = device,
                        n_jobs = n_jobs,
                        )
                    model.fit(
                        X, y,
                        eval_set = [(X, y)],
                        verbose = False if verbose<1 else True,
                        )
                else: #default

                    params = {

                        #core
                        'booster': 'gbtree',
                        'enable_categorical': True,
                        'grow_policy': 'lossguide', 'max_depth': 0,
                        'max_leaves': max(len(X) // 10000, 10),
                        'max_delta_step': 0,

                        #weighted newton step
                        'objective': 'multi:softmax',
                        'reg_lambda': 10,

                        #colsample
                        'colsample_bytree': 0.5,
                        'colsample_bylevel': 1,
                        'colsample_bynode': 1,

                        #stop
                        'eval_metric': 'mlogloss',
                        'early_stopping_rounds': 10,

                        #random
                        'random_state': random_seed,

                        #device
                        'tree_method': 'hist',
                        'device': device,
                        'n_jobs': n_jobs,

                        #descent
                        'learning_rate': 0.01,
                        'n_estimators': 1000+1,

                        }
                    params.update(config.get('classifier', {})) #parameters update
                    model = XGBClassifier(**params)

                    #distribution
                    if verbose < 1:
                        pass
                    else:
                        temp = np.unique(y, return_counts = True)[1].tolist() #class distribution
                        print(f"\
Target: {i}\n\
encoder: {encoder}\n\
#classes: {len(temp)}\n\
distribution: {temp}\
                            ")

                    if p_eval == 0:
                        model.fit(
                            X, y,
                            eval_set = [(X, y)],
                            verbose = 0 if verbose<1 else (params['n_estimators'] - 1) // verbose,
                            )
                    else:

                        # At least one instance per class is contained in the fit.
                        idx_unique = np.unique(y, return_index = True)[1]
                        X_, X_eval, y_, y_eval = train_test_split(
                            X[np.delete(np.arange(len(X)), idx_unique, axis = 0), :],
                            np.delete(y, idx_unique, axis = 0),
                            shuffle = True,
                            test_size = p_eval,
                            random_state = random_seed,
                            )
                        X_ = pl.concat([X_, X[idx_unique]], how = 'vertical')
                        y_ = np.concatenate([y_, y[idx_unique]], axis = 0)
                        model.fit(
                            X_, y_,
                            eval_set = [(X_, y_), (X_eval, y_eval)],
                            verbose = 0 if verbose<1 else (params['n_estimators'] - 1) // verbose,
                            )



                #probability complement
                pred_prob = model.predict_proba(X).astype('float64')
                error = 1. - pred_prob[np.arange(len(pred_prob)), y]

                #stored
                self.trainerrors[i] = error
                self.ensemble[i] = {
                    'type': 'classifier',
                    'encoder-target': encoder,
                    'model': model,
                    }

            else:
                y = scaler.transform(y.reshape([y.shape[0], -1]))[:, 0]

                #different base-learners
                if self._learner == 'RF':
                    model = RandomForestRegressor(min_samples_leaf = min(1000, len(X)//100+1), random_seed = random_seed, n_jobs = n_jobs)
                    model.fit(
                        X, y,
                        category_columns = [c for c in self._processor.categories.keys() if c != i],
                        )
                elif self._learner == 'MLP':
                    model = MLPRegressor(random_seed = random_seed)
                    model.fit(
                        X, y,
                        category_columns = [c for c in self._processor.categories.keys() if c != i],
                        )
                elif self._learner == 'rf':
                    model = XGBRFRegressor(
                        min_child_weight = min(1000., len(X)/100) * 1.,
                        subsample = 0.5, #ratio matching the mean stability of bootstrap
                        colsample_bytree = 1,
                        colsample_bylevel = 1,
                        colsample_bynode = sqrt(len(X.columns)) / float(len(X.columns)),
                        max_depth = 0,
                        enable_categorical = True,
                        objective = 'reg:squarederror',
                        eval_metric = 'rmse',
                        tree_method = 'hist',
                        random_state = random_seed,
                        device = device,
                        n_jobs = n_jobs,
                        )
                    model.fit(
                        X, y,
                        eval_set = [(X, y)],
                        verbose = False if verbose<1 else True,
                        )
                else: #default

                    params = {

                        #core
                        'booster': 'gbtree',
                        'enable_categorical': True,
                        'grow_policy': 'lossguide', 'max_depth': 0,
                        'max_leaves': max(len(X) // 10000, 10),
                        'max_delta_step': 1,

                        #weighted newton step
                        'objective': 'reg:pseudohubererror',
                        'reg_lambda': 10,

                        #colsample
                        'colsample_bytree': 0.5,
                        'colsample_bylevel': 1,
                        'colsample_bynode': 1,

                        #stop
                        'eval_metric': 'mphe',
                        'early_stopping_rounds': 10,

                        #random
                        'random_state': random_seed,

                        #device
                        'tree_method': 'hist',
                        'device': device,
                        'n_jobs': n_jobs,

                        #descent
                        'learning_rate': 0.01,
                        'n_estimators': 1000+1,

                        }
                    params.update(config.get('regressor', {})) #parameters update
                    model = XGBRegressor(**params)

                    #distribution
                    if verbose < 1:
                        pass
                    else:
                        print(f"\
Target: {i}\n\
scaler: {scaler}\n\
minmax: {np.quantile(y, [0, 1], axis = 0)}\n\
median: {np.median(y, axis = 0)}\n\
IQR: {np.quantile(y, [0.25, 0.75], axis = 0)}\n\
mean: {y.mean(axis = 0, dtype = 'float64')}\n\
stddev: {y.std(axis = 0, dtype = 'float64')}\
                            ")

                    if p_eval == 0:
                        model.fit(
                            X, y,
                            eval_set = [(X, y)],
                            verbose = 0 if verbose<1 else (params['n_estimators'] - 1) // verbose,
                            )
                    else:

                        X_, X_eval, y_, y_eval = train_test_split(X, y, shuffle = True, test_size = p_eval, random_state = random_seed)
                        model.fit(
                            X_, y_,
                            eval_set = [(X_, y_), (X_eval, y_eval)],
                            verbose = 0 if verbose<1 else (params['n_estimators'] - 1) // verbose,
                            )



                #range-normalized error
                pred = model.predict(X).astype('float64')
                error = pred - y
                scale = (y.max(axis = 0) - y.min(axis = 0)).tolist()
                error = error / scale #train-normalized

                #stored
                self.trainerrors[i] = error
                self.ensemble[i] = {
                    'type': 'regressor',
                    'scaler-target': scaler,
                    'model': model,
                    'scale-error': scale,
                    }


        self.trainerrors = pl.DataFrame(self.trainerrors)


    def predict(self, param_df, train_fpr = 1e-4, return_ranks = False, return_scores = False, return_residuals = False):
        if not isinstance(param_df, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(train_fpr, (float, int)):
            raise TypeError('\'train_fpr\' should be real.')
        if not isinstance(return_ranks, bool):
            raise TypeError('\'return_ranks\' should be boolean.')
        if not isinstance(return_scores, bool):
            raise TypeError('\'return_scores\' should be boolean.')
        if not isinstance(return_residuals, bool):
            raise TypeError('\'return_residuals\' should be boolean.')
        df = param_df.clone(); del param_df
        returns = []

        df = self._processor.transform(df)

        errors = {}
        for i in tqdm(self.ensemble.keys()):
            label = df[:, i].to_numpy(writable = True)
            if label.dtype == 'object':
                label = label.astype('str')
            data = df.drop([i])

            if self.ensemble[i]['type'] == 'indicator':

                err = (label != self.ensemble[i]['value']).astype('float64') #unseens assigned 1

                errors[i] = err

            elif self.ensemble[i]['type'] == 'classifier':
                label = self.ensemble[i]['encoder-target'].transform(label.reshape([label.shape[0], -1]))[:, 0]

                #unseens located
                temp = self.ensemble[i]['model'].classes_
                location = label.reshape([label.shape[0], -1]) == temp.reshape([-1, temp.shape[0]])
                unseen = ~np.logical_or.reduce(location, axis = 1) #indices of unseens

                #probability complement
                pred_prob = self.ensemble[i]['model'].predict_proba(data).astype('float64')
                err_seen = 1. - pred_prob[location]
                err_unseen = np.float64(1) #unseens assigned 1
                err = np.zeros(label.shape, dtype = 'float64')
                err[~unseen] = err_seen
                err[unseen] = err_unseen

                errors[i] = err

            else:
                label = self.ensemble[i]['scaler-target'].transform(label.reshape([label.shape[0], -1]))[:, 0]

                #range-normalized error
                pred = self.ensemble[i]['model'].predict(data)
                err = pred - label
                err = err / self.ensemble[i]['scale-error'] #train-normalized  ## Outbounds assigned greater than 1?

                errors[i] = err


        errors = pl.DataFrame(errors)

        #mean along columns
        trainerror_absmean = np.absolute(self.trainerrors.to_numpy(writable = False)).mean(axis = 1, dtype = 'float64')
        error_absmean = np.absolute(errors.to_numpy(writable = False)).mean(axis = 1, dtype = 'float64')

        #threshold
        threshold = np.quantile(
            trainerror_absmean,
            1. - train_fpr,
            axis = 0,
            ).tolist()

        #predicted
        prediction = error_absmean >= threshold
        returns.append(prediction)

        # The reason I put the ranking first is that sometimes only ranking but scores is defined.
        if return_ranks: #ranking
            rank = stats.rankdata(error_absmean, method = 'min', axis = 0)
            returns.append(rank)
        if return_scores: #scores
            returns.append(error_absmean)

        #dependency-space transform
        if return_residuals:
            returns.append(errors)

        if len(returns) == 0:
            return None
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns


    def xpredict(self, param_df, train_fpr = 0.0001, return_detail = False, return_ranks = False):
        if not isinstance(param_df, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(train_fpr, (float, int)):
            raise TypeError('\'train_fpr\' should be real.')
        if not isinstance(return_detail, bool):
            raise TypeError('\'return_detail\' should be boolean.')
        if not isinstance(return_ranks, bool):
            raise TypeError('\'return_ranks\' should be boolean.')
        df = param_df.clone(); del param_df
        returns = []

        df = self._processor.transform(df)


        # Thresholds are searched everytime this function is called.
        # At first glance, it may seem the validation information is used when the validation data is the argument.
        # However, no outside information is used as the losses sweeped during the search is of the train.
        # - thresholds -

        trainerror = self.trainerrors.to_numpy(writable = True)
        threshold_sweep = np.unique(trainerror, axis = 0)
        threshold_sweep.sort(axis = 0) #alignment broken; independent sort
        threshold_sweep = np.insert(
            threshold_sweep,
            threshold_sweep.shape[0],
            threshold_sweep[-1] + np.spacing(threshold_sweep[-1]),
            axis = 0,
            )
        threshold_sweep = np.flip(threshold_sweep, axis = 0) #descending order
    
        trainfpr = []
        for idx, i in enumerate(threshold_sweep):
    
            detection = trainerror >= i.reshape([-1, i.shape[0]])
            detection = np.logical_or.reduce(detection, axis = 1)
            fpr = np.unique(detection, axis = 0, return_counts = True)[1].tolist()
    
            if len(fpr) == 1:
                if detection[0] == True: # It is not python-True but numpy-True.
                    fpr = 1
                else:
                    fpr = 0
            else:
                fpr = fpr[1] / len(trainerror)
    
            trainfpr.append(fpr)
            if fpr > train_fpr: # The equality has been removed to consider the case "train_fpr == 0".
                before = threshold_sweep[idx - 1]
                after = threshold_sweep[idx]
                break
    
        before_fpr = trainfpr[-2]
        after_fpr = trainfpr[-1]
        coef = (train_fpr - before_fpr) / (after_fpr - before_fpr)
    
        temp = (before + np.float64(coef) * (after - before)).tolist()
        thresholds = {}
        for idx, i in enumerate(self.trainerrors.columns):
            thresholds[i] = temp[idx]


        #ranking
        if return_ranks:
            rank = []

        detections = {}
        for i in tqdm(self.ensemble.keys()):
            label = df[:, i].to_numpy(writable = True)
            if label.dtype == 'object':
                label = label.astype('str')
            data = df.drop([i])

            if self.ensemble[i]['type'] == 'indicator':

                detections[i] = label != self.ensemble[i]['value']

                #ranking
                if return_ranks:
                    rnk = np.zeros(label.shape, dtype = 'int64') #lowest ranking to the singular
                    rnk[label != self.ensemble[i]['value']] = len(label) + 1 #highest ranking to the unseen
                    rank.append(rnk)

            elif self.ensemble[i]['type'] == 'classifier':
                label = self.ensemble[i]['encoder-target'].transform(label.reshape([label.shape[0], -1]))[:, 0]

                #negative-log error
                pred_prob = self.ensemble[i]['model'].predict_proba(data).astype('float64')
                temp = self.ensemble[i]['model'].classes_
                location = label.reshape([label.shape[0], -1]) == temp.reshape([-1, temp.shape[0]])
                unseen = ~np.logical_or.reduce(location, axis = 1)
                error_seen = -np.log(pred_prob[location], dtype = 'float64')
                detections[i] = np.zeros(label.shape, dtype = 'bool') #empty prepared

                #ranking
                if return_ranks:
                    rnk = np.zeros(label.shape, dtype = 'int64')
                    rnk[~unseen] = stats.rankdata(error_seen, method = 'min', axis = 0) #normal ranking
                    rnk[unseen] = len(label) + 1 #highest ranking to the unseen
                    rank.append(rnk)

                detections[i][~unseen] = error_seen >= thresholds[i]
                detections[i][unseen] = True

            else:
                label = self.ensemble[i]['scaler-target'].transform(label.reshape([label.shape[0], -1]))[:, 0]

                #absolute relative error
                pred = self.ensemble[i]['model'].predict(data)
                error = np.absolute(pred - label)
                error = error / self.ensemble[i]['scale-error'] #train-normalized

                #ranking
                if return_ranks:
                    rnk = stats.rankdata(error, method = 'min', axis = 0) #normal ranking
                    rank.append(rnk)

                detections[i] = error >= thresholds[i]


        detections = pl.DataFrame(detections)
        detection = np.logical_or.reduce(detections.to_numpy(writable = True), axis = 1)

        if return_ranks:
            rank = np.stack(rank, axis = 1)
            rank = rank.max(axis = 1)

        returns.append(detection)
        if return_detail:
            returns.append(detections)
        if return_ranks:
            returns.append(rank)

        if len(returns) < 1:
            return None
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns
