from common import *
import logging
logger = logging.getLogger(name = __name__)


## Needs be order-agnostic.
class Pipe:
    def __init__(self):
        pass

    def __repr__(self):
        return 'pipeline'

    def fit(self, train:pl.DataFrame, categories:dict, Scaler:type, Encoder:type)->None:
        assert isinstance(train, pl.DataFrame), 'wrong type'
        assert isinstance(categories, dict), 'wrong type'
        assert isinstance(Scaler, type), 'wrong type'
        assert isinstance(Encoder, type), 'wrong type'
        self._categories = copy(categories)
        self._columns = train.columns

        #numerical
        arr_numerical = train.drop(list(self._categories.keys())).to_numpy(writable = True)
        self.scaler = Scaler()
        self.scaler.fit(arr_numerical)

        #categorical with cases
        if len(self._categories) < 1:
            pass
        else:

            arr_categorical = train.select(list(self._categories.keys())).to_numpy(writable = True).astype('str')
            self.encoder = Encoder(
                categories = [c.categories.to_list() for c in self._categories.values()],
                handle_unknown = 'error',
                dtype = np.int64,
                encoded_missing_value = -1,
                )
            self.encoder.fit(arr_categorical)



    def transform(self, df:pl.DataFrame)->pl.DataFrame:
        assert isinstance(df, pl.DataFrame), 'wrong type'

        arr_numerical = df.drop(list(self._categories.keys())).to_numpy(writable = True)
        transformed_numerical = self.scaler.transform(arr_numerical)
        columns_numerical = [c for c in self._columns if c not in self._categories.keys()]

        #categorical with cases
        if len(self._categories) < 1:
            transformed = pl.DataFrame(transformed_numerical, orient = 'row', schema = columns_numerical)
        else:

            arr_categorical = df.select(list(self._categories.keys())).to_numpy(writable = True).astype('str')
            transformed_categorical = self.encoder.transform(arr_categorical)
            columns_categorical = list(self._categories.keys())

            transformed = pl.concat([
                pl.DataFrame(transformed_numerical, orient = 'row', schema = columns_numerical),
                pl.DataFrame(transformed_categorical, orient = 'row', schema = columns_categorical),
                ], how = 'horizontal')

        #feature order aligned
        transformed = transformed.select(self._columns)

        return transformed


    def inverse_transform(self, df:pl.DataFrame)->pl.DataFrame:
        assert isinstance(df, pl.DataFrame), 'wrong type'

        #numerical
        arr_numerical = df.drop(list(self._categories.keys())).to_numpy(writable = True)
        original_numerical = self.scaler.inverse_transform(arr_numerical)
        columns_numerical = [c for c in self._columns if c not in self._categories.keys()]

        #categorical with cases
        if len(self._categories) < 1:
            original = pl.DataFrame(original_numerical, orient = 'row', schema = columns_numerical)
        else:

            arr_categorical = df.select(list(self._categories.keys())).to_numpy(writable = True)
            original_categorical = self.encoder.inverse_transform(arr_categorical)
            columns_categorical = list(self._categories.keys())

            original = pl.concat([
                pl.DataFrame(original_numerical, orient = 'row', schema = columns_numerical),
                pl.DataFrame(original_categorical, orient = 'row', schema = columns_categorical),
                ], how = 'horizontal')

        #back to enum
        original = original.cast(self._categories, strict = True)

        #feature order aligned
        original = original.select(self._columns)

        return original
