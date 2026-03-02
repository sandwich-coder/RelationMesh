import logging
logger = logging.getLogger(name = 'refine')
import polars as pl
import pandas as pd
from catboost.datasets import higgs

SOURCE = 'catboost'


if SOURCE == 'csv': #website download
    df = pl.read_csv('raw.csv', has_header = False, infer_schema = False)

if SOURCE == 'catboost': #catboost download
    pdf_1, pdf_2 = higgs()
    df = pd.concat([pdf_1, pdf_2], copy = True, axis = 'index', ignore_index = True)
    df = pl.from_pandas(df)

df = df.select(df.columns[1:] + df.columns[0:1])
df_columns = []
for i in range(len(df.columns))[:-1]:
    df_columns.append(f'col{i + 1}')
df_columns.append('label')
df.columns = df_columns

if SOURCE == 'csv':
    df = df.cast({pl.String: pl.Float64}).cast({'label': pl.Boolean})
elif SOURCE == 'catboost':
    df = df.cast({pl.Float32: pl.Float64}).cast({'label': pl.Boolean})

if df.null_count().sum_horizontal(ignore_nulls = False).item() > 0:
    logger.info('Nulls are processed.')

df.write_parquet('data.parquet', use_pyarrow = True)
