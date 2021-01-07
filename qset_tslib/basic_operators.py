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


def constant(value, prototype):
    """
    :param value: float
    :param prototype:
    :return: constant matrix with the shape of the prototype
    """
    res = prototype.copy()
    res[:] = value
    return res


def df_max(df1, df2):
    nan_mask = reduce(lambda df1, df2: df1 | df2, [df.isnull() for df in [df1, df2]])
    res = ifelse(df1 > df2, df1, df2)
    if nan_mask is not None:
        res = res.where(~nan_mask)
    return res


def df_min(df1, df2):
    nan_mask = reduce(lambda df1, df2: df1 | df2, [df.isnull() for df in [df1, df2]])
    res = ifelse(df1 < df2, df1, df2)
    if nan_mask is not None:
        res = res.where(~nan_mask)
    return res


def test_df_max_and_df_min():
    df1 = pd.DataFrame(np.arange(0, 5, 1))
    df2 = df1.sort_values(by=0, ascending=False).reset_index(drop=True)  # reversed df
    print(df_max(df1, df2))
    print(df_min(df1, df2))


if __name__ == '__main__':
    test_df_max_and_df_min()