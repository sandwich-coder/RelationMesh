import logging
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(name = 'refine-merge')
import polars as pl

categorical = []


df = pl.concat([
    pl.read_csv('training-flow.csv', infer_schema = False),
    pl.read_csv('test-flow.csv', infer_schema = False),
    ])
df = df.drop([
    'attack_name',

    #non-behavioral
    'destination port',

    ])
df = df.rename({'attack_flag': 'label', 'attack_step': 'kind'})

dtypes = {}
for i in [c for c in df.columns if c not in (['label', 'kind'] + categorical)]:
    dtypes[i] = pl.Float64
df = df.cast(dtypes)
df = df.cast({'label': pl.Int64}).cast({'label': pl.Boolean})

if df.null_count().to_numpy().sum(dtype = 'int64').tolist() != 0:
    logger.info('Nulls are processed.')
    #?fill_null
    df = df.drop_nulls()
#?drop_nans

df.write_parquet('data.parquet')
