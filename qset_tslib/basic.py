import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce


def ifelse(cond, df, other=np.nan, **kwargs):
    return df.where(cond, other=other, **kwargs)


def test_ifelse():
    df1 = pd.DataFrame(np.arange(0, 5, 1))
    df2 = df1.sort_values(by=0, ascending=False).reset_index(drop=True)  # reversed df
    print(ifelse(df1 > df2, df1, df2))
    print(ifelse(df1 < df2, df1, df2))


def mask(df, condition):
    return df.where(condition)


def df_max(df1, df2):
    nan_mask = reduce(lambda df1, df2: df1 | df2, [df.isnull() for df in [df1, df2]])
    res = ifelse(df1 > df2, df1, df2)
    res = res.where(~nan_mask)
    return res


def df_min(df1, df2):
    nan_mask = reduce(lambda df1, df2: df1 | df2, [df.isnull() for df in [df1, df2]])
    res = ifelse(df1 < df2, df1, df2)
    res = res.where(~nan_mask)
    return res


def test_df_max_and_df_min():
    df1 = pd.DataFrame(np.arange(0, 5, 1))
    df2 = df1.sort_values(by=0, ascending=False).reset_index(drop=True)  # reversed df
    print(df_max(df1, df2))
    print(df_min(df1, df2))


def agg(df, by=None, func=None, level=None, **kwargs):
    return df.groupby(by=by, level=level, **kwargs).agg(func)


def rank(df, a=0., b=1., axis=1):
    """ A simplified version of pd.DataFrame.rank with specified delimiters. """
    return df.rank(axis=axis, pct=True) * (b - a) + a



if __name__ == '__main__':
    test_df_max_and_df_min()