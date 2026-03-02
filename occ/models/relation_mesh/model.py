from common import *
import logging
logger = logging.getLogger(name = __name__)
from xgboost import (

    XGBRegressor,
    XGBClassifier,

    XGBRFRegressor,
    XGBRFClassifier,

    )
from sklearn.preprocessing import (

    #numerical target
    RobustScaler,
    StandardScaler, #random-forest

    #categorical target
    OrdinalEncoder,

    )
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import randint
from math import sqrt
import platform
from psutil import cpu_count
import tempfile

from ..processor import Processor
from .pipe import Pipe
from ._learners import (
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
threads = cpus - 1
if platform.system() == 'Linux':
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    if 'GenuineIntel' in cpuinfo: #server
        threads = 2
elif platform.system() == 'Darwin': #local
    threads = max(threads - 4, 1)
logger.info(f"#threads: {threads}")


# - functions -

def _soften(df:pl.DataFrame, param_one_columns:list)->pl.DataFrame:
    one_columns = copy(param_one_columns); del param_one_columns

    rows = len(df)
    one_columns = list(set(one_columns))

    updated = {}
    for i in one_columns:
        updated[i] = np.ones([rows], dtype = 'float64')
    softened = df.with_columns(**updated)

    return softened



class RelationMesh:
    def __init__(self, base_learner = 'default'):
        if not isinstance(base_learner, str):
            raise TypeError('\'base_learner\' should be a string.')
        self._processor = Processor()
        self._learner = base_learner
        self.pipe = Pipe()

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

    def fit_(
        self,
        target,
        train,
        train_,
        config,
        p_eval,
        random_seed,
        verbose,
        ):
        #typechecked at the caller

        X = train.drop([target])
        y = train_[target].to_numpy(writable = True)

        #trivial case
        if target in self._processor.constants.keys():
            logger.info(f"The target \'{target}\' is constant, falling back to rule-based indicator.")
            model = None

        elif target in self._processor.categorical:

            #different base-learners
            if self._learner == 'RF':
                model = RandomForestClassifier(min_samples_leaf = min(1000, len(X)//100+1), random_seed = random_seed, n_jobs = n_jobs)
                model.fit(
                    X, y,
                    category_columns = [c for c in self._processor.categories.keys() if c != target],
                    )
            elif self._learner == 'MLP':
                model = MLPClassifier(random_seed = random_seed)
                model.fit(
                    X, y,
                    category_columns = [c for c in self._processor.categories.keys() if c != target],
                    )
            elif self._learner == 'TabPFN':
                logger.error('The TabPFN variant is in progress.')
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
                    n_jobs = threads,
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
                    'n_jobs': threads,

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
Target: {target}\n\
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



        else: #numerical

            #different base-learners
            if self._learner == 'RF':
                model = RandomForestRegressor(min_samples_leaf = min(1000, len(X)//100+1), random_seed = random_seed, n_jobs = n_jobs)
                model.fit(
                    X, y,
                    category_columns = [c for c in self._processor.categories.keys() if c != target],
                    )
            elif self._learner == 'MLP':
                model = MLPRegressor(random_seed = random_seed)
                model.fit(
                    X, y,
                    category_columns = [c for c in self._processor.categories.keys() if c != target],
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
                    n_jobs = threads,
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
                    'n_jobs': threads,

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
Target: {target}\n\
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




        return {target: model}


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

        #processor fitted
        self._processor.fit(train, categorical, other)

        # This step is not just a category reduction.
        # The polars 'enum' encoding is a part of the training,
        # which means any categorical data from outside should be given as pure strings.
        #prepared
        train = self._processor.transform(train)
        logger.info(f"Training Size: {train.shape} | %eval: {100 * p_eval:.2g}%")

        #pipeline fitted
        self.pipe.fit(
            train,
            self._processor.categories,
            StandardScaler if self._learner!='default' else config.get('Scaler', RobustScaler),
            OrdinalEncoder if self._learner!='default' else config.get('Encoder', OrdinalEncoder),
            )

        ## It turns out that xgb.XGBRegressor has argument 'feature_types'.
        ## If that overrides whatever the dataframe says, I don't have to store the feature dataframe.
        ## It will be much desirable to use only the "piped", where the categories exist in integers,
        ## and transform back to the original space once the reconstruction is done.
        ## Yet, it needs to be checked whether the result remains unchanged.
        #targets
        train_ = self.pipe.transform(train)


        # - train -

        models = []
        for i in train.columns:
            models.append(
                self.fit_(i, train, train_, config, p_eval, random_seed, verbose),
                )


        #models stored
        self.models = {}
        for i in models:
            self.models.update(i)


        # - range-normalized error -

        #reconstruct
        train_soft = _soften(
            train,
            list(self._processor.constants.keys()) + categorical,
            ).to_numpy(writable = True)
        rec_soft = self.soft_reconstruct(train).to_numpy(writable = True)

        #scale
        self._trainscale = train_soft.max(axis = 0) - train_soft.min(axis = 0)
        self._trainscale[self._trainscale == 0] = 1

        #scores
        error = (rec_soft - train_soft) / np.expand_dims(self._trainscale, 0)
        self.trainscore = np.absolute(error).mean(axis = 1)



    def reconstruct(self, param_df):
        if not isinstance(param_df, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        df = param_df.clone(); del param_df

        #prepared
        df = self._processor.transform(df)

        reconstructed = {}
        for i in tqdm(self.models.keys(), ncols = 70):
            target = i
            model = self.models[i]

            if target in self._processor.constants.keys(): #trivial
                reconstructed[target] = np.full(
                    [len(df)],
                    self._processor.constants[target],
                    dtype = 'float64',
                    )
            else:
                reconstructed[target] = model.predict(df.drop([target]))

        reconstructed = pl.DataFrame(reconstructed).cast({pl.Int32: pl.Int64, pl.Float32: pl.Float64}) #aggregate
        reconstructed = self.pipe.inverse_transform(reconstructed) #original scale

        return reconstructed


    def soft_reconstruct(self, param_df):
        if not isinstance(param_df, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        df = param_df.clone(); del param_df

        #prepared
        df = self._processor.transform(df) #features
        df_ = self.pipe.transform(df) #targets

        #reconstructed
        rec_trivial = {} #feature-space
        rec_categorical = {} #feature-space
        rec_numerical_ = {} #target-space
        for i in tqdm(self.models.keys(), ncols = 70):
            target = i
            model = self.models[i]

            if target in self._processor.constants.keys(): #basically a binary classification problem

                # Regardless of datatype, the model outputs 100% chance for the stored constant
                # and 0% otherwise.
                rec_trivial[target] = df[target].to_numpy() == self._processor.constants[target]
                rec_trivial[target] = rec_trivial[target].astype('float64')

            elif target in self._processor.categorical: #predicted probability of the label

                is_unseen = df_[target].to_numpy() > model.classes_.max(axis = 0) # The xgboost accepts the class indices only in the smallest natural numbers.
                probs = model.predict_proba(df.drop([target]))
                prob = np.zeros([len(probs)], dtype = 'float64')

                temp = df_[target].to_numpy()[~is_unseen]
                prob[~is_unseen] = probs[~is_unseen][range(len(temp)), temp]
                prob[is_unseen] = 0

                # the probability of predicting the given label
                rec_categorical[target] = prob

            else:

                rec_numerical_[target] = model.predict(df.drop([target]))


        df_re = self.pipe.inverse_transform(
            df_.with_columns(**rec_numerical_),
            ) #numerical
        df_re = df_re.with_columns(**rec_trivial) #trivial
        df_re = df_re.with_columns(**rec_categorical) #categorical
        df_re = df_re.cast({pl.Float32: pl.Float64})

        return df_re


    def predict(self, param_df, train_fpr = 1e-4, return_ranks = False, return_scores = False):
        if not isinstance(param_df, pl.DataFrame):
            raise TypeError(f"Input should be a polars.DataFrame. Got {type(param_df)}.")
        if not isinstance(train_fpr, float | int):
            raise TypeError('\'train_fpr\' should be a real.')
        if not isinstance(return_ranks, bool):
            raise TypeError('\'return_ranks\' should be boolean.')
        if not isinstance(return_scores, bool):
            raise TypeError('\'return_scores\' should be boolean.')
        df = param_df.clone(); del param_df
        returns = []

        #prepared
        df = self._processor.transform(df)


        # - range-normalized error -

        #reconstruct
        ori = _soften(
            df,
            list(self._processor.constants.keys()) + self._processor.categorical,
            ).to_numpy(writable = True) #original
        rec = self.soft_reconstruct(df).to_numpy(writable = True) #reconstructed

        #scores
        error = (rec - ori) / np.expand_dims(self._trainscale, 0)
        score = np.absolute(error).mean(axis = 1)


        #prediction
        threshold = np.quantile(self.trainscore, 1 - train_fpr, axis = 0).tolist()
        prediction = score >= threshold

        #ranking
        rank = stats.rankdata(score, method = 'min', axis = 0)

        returns.append(prediction)
        if return_ranks:
            returns.append(rank)
        if return_scores:
            returns.append(score)

        #returns
        if len(returns) == 0:
            pass
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns
