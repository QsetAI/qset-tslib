import numpy as np
import pandas as pd

from functools import reduce

import qset_tslib as tslib


def max(df1, df2):
    nan_mask = reduce(lambda df1, df2: df1 | df2, [df.isnull() for df in [df1, df2]])
    res = tslib.ifelse(df1 > df2, df1, df2)
    res = res.where(~nan_mask)
    return res


def min(df1, df2):
    nan_mask = reduce(lambda df1, df2: df1 | df2, [df.isnull() for df in [df1, df2]])
    res = tslib.ifelse(df1 < df2, df1, df2)
    res = res.where(~nan_mask)
    return res


def sign(df, *args, **kwargs):
    return np.sign(df, *args, **kwargs)


def log(df, *args, **kwargs):
    return np.log(df, *args, **kwargs)


def exp(df, *args, **kwargs):
    return np.exp(df, *args, **kwargs)


def signed_power(df, pow):
    return np.sign(df) * df.abs().pow(pow)


def power(df, n):
    return df.pow(n)


def abs(df):
    return df.abs()


def test_max_min():
    df1 = pd.DataFrame(np.arange(0, 5, 1))
    df2 = df1.sort_values(by=0, ascending=False).reset_index(drop=True)  # reversed df
    print(max(df1, df2))
    print(min(df1, df2))


if __name__ == "__main__":
    test_max_min()
