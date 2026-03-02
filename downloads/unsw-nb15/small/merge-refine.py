import logging
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(name = 'refine-merge')
import polars as pl

categorical = [
    'proto',
    'service',
    'state',
    ]


df = pl.concat([
    pl.read_csv('train.csv', infer_schema = False),
    pl.read_csv('test.csv', infer_schema = False),
    ])
df = df.drop([

    #non-behavioral
    'id',

    ])
df = df.rename({'attack_cat': 'kind'})
df = pl.concat([df.drop(['kind']), df.select(['kind'])], how = 'horizontal') #reordered

dtypes = {}
for i in [c for c in df.columns if c not in (['label', 'kind'] + categorical)]:
    dtypes[i] = pl.Float64
df = df.cast(dtypes)
df = df.cast({'label': pl.Int64}).cast({'label': pl.Boolean})

if df.null_count().to_numpy().sum(dtype = 'int64').tolist() > 0:
    logger.info('nulls processed')
    #?fill_null
    df = df.drop_nulls()
#?drop_nans

df.write_parquet('data.parquet')
