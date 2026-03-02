import logging
logging.basicConfig(level = 'INFO')
import numpy as np
import polars as pl

categorical = [
    'proto',
    'service',
    'state',
    ]


columns = pl.read_csv('NUSW-NB15_features.csv', infer_schema = False, encoding = 'utf8-lossy')

df = pl.concat([
    pl.scan_csv('UNSW-NB15_1.csv', infer_schema = False, has_header = False),
    pl.scan_csv('UNSW-NB15_2.csv', infer_schema = False, has_header = False),
    pl.scan_csv('UNSW-NB15_3.csv', infer_schema = False, has_header = False),
    pl.scan_csv('UNSW-NB15_4.csv', infer_schema = False, has_header = False),
    ], how = 'vertical').collect(engine = 'streaming')
df.columns = columns[:, 'Name'].to_list()

df = df.rename({'Label': 'label', 'attack_cat': 'kind'})
df = pl.concat([
    df.drop(['kind']),
    df.select(['kind']),
    ], how = 'horizontal')

df = df.drop([

    #non-behavioral
    'Stime',
    'Ltime',
    'srcip',
    'sport',
    'dstip',
    'dsport',

    ])

#nulls
if df.null_count().to_numpy(writable = False).squeeze().sum(axis = 0).tolist() > 0:
    logging.info('nulls processed')

    df = df.with_columns(
        kind = pl.col('kind').fill_null('-'),
        ct_flw_http_mthd = pl.col('ct_flw_http_mthd').fill_null('0'),
        is_ftp_login = pl.col('is_ftp_login').fill_null('0'),
        )


#nonsense
df = df.with_columns(
    is_ftp_login = pl.col('is_ftp_login').replace({'2': '1', '4': '1'}),
    ct_ftp_cmd = pl.col('ct_ftp_cmd').replace({' ': '0'}),
    )

dtypes = {}
for i in [c for c in df.columns if c not in (['label', 'kind'] + categorical)]:
    dtypes[i] = pl.Float64
df = df.cast(dtypes)
df = df.cast({'label': pl.Int64}).cast({'label': pl.Boolean})

#saved
df.write_parquet('data.parquet')
