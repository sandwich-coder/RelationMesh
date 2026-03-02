import logging
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(name = 'refine-merge')
import polars as pl
# 'null' denotes "missing"
# 'nan' denotes the value called 'nan'.

categorical = [
    'protocol',
    'service',
    'flag',
    ]


# Nulls are not first-removed, as they may denote something.
df = pl.concat([
    pl.read_csv('training-flow.csv', infer_schema = False),
    pl.read_csv('test-flow.csv', infer_schema = False),
    ])

dtypes = {}
for i in [c for c in df.columns if c not in (['attack_name', 'attack_step', 'attack_flag'] + categorical)]:
    dtypes[i] = pl.Float64
df = df.cast(dtypes)

if df.null_count().to_numpy().sum(dtype = 'int64').tolist() != 0:
    logger.info('Nulls are processed.')
    #?fill_null
    df = df.drop_nulls()
df = df.drop(['unknown'])
#?drop_nans

df = df.drop(['attack_name'])
df = df.rename({'attack_flag': 'label', 'attack_step': 'kind'})
df = df.cast({'label': pl.Int64}).cast({'label': pl.Boolean})

df.write_parquet('data.parquet')
