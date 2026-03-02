from common import *
import logging
logger = logging.getLogger(name = __name__)
from sklearn.preprocessing import OneHotEncoder

def _onehot(param_df, categories):
    df = param_df.clone(); del param_df
    if len(categories) < 1:
        return df.to_numpy(writable = True)
    df = df.cast({pl.Enum: pl.String})

    df_oh_numerical = df.drop(list(categories.keys())).to_numpy(writable = True)

    # The enoder by default scans the categories on fit but only considers what are there, not able to encode spare slots.
    # Also, the default behavior encodes the classes in ascii order and the categories should be given manually to use the desired order.
    ohe = OneHotEncoder(
        categories = [categories[c].categories.to_list() for c in categories.keys()],
        sparse_output = False,
        dtype = 'bool',
        )
    df_oh_categorical = ohe.fit_transform(
        df.select(list(categories.keys())),
        )

    df_oh = np.concatenate([
        df_oh_numerical,
        df_oh_categorical.astype('float64'),
        ], axis = 1)
    return df_oh



class Processor:
    def __repr__(self):
        return 'processor'

    def __init__(self):
        pass

    def fit(self, df, categorical, other):
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Data should be a polars.DataFrame. Got {type(df)}")
        if not isinstance(categorical, list):
            raise TypeError(f"Category columns should be a list. Got {type(categorical)}")
        if not isinstance(other, str):
            raise TypeError(f"Name of the spare class should be a string. Got {type(other)}")
        self._other = other

        #category reduced
        self.categories = {}
        for i in categorical:
            frequents = df[:, i].value_counts(sort = True, name = 'N')[:, i].head(29).to_list() #most frequent
            frequents.append(self._other)
            self.categories[i] = pl.Enum(frequents)

        #stored for external use
        self.columns = df.columns
        self.categorical = [c for c in self.columns if c in self.categories.keys()]
        self.numerical = [c for c in self.columns if c not in self.categorical]

        #trivial columns
        constant = df.select(pl.all().n_unique() == 1).to_numpy(writable = True).squeeze(axis = 0)
        constant = np.array(df.columns)[constant].tolist()
        self.constants = {}
        for i in constant:
            self.constants[i] = df[i][0]


    def transform(self, param_df, onehot = False):
        if not isinstance(param_df, pl.DataFrame):
            raise TypeError('Input should be a polars.DataFrame.')
        if not isinstance(onehot, bool):
            raise TypeError('\'onehot\' should be boolean.')
        df = param_df.clone(); del param_df

        #cast to the stored categories
        df = df.cast(self.categories, strict = False)

        #merge nulls into the spare
        df = df.cast({pl.Enum: pl.String})
        df = df.fill_null(self._other)
        df = df.cast(self.categories, strict = True)

        if onehot:
            df = _onehot(df, self.categories)
        return df
