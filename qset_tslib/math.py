import numpy as np
import pandas as pd

from qset_tslib.dataseries import ifelse


def sign(df):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: sign of data as int: -1, 0 or 1
    """
    return np.sign(df)


def log(df):
    return np.log(df)


def exp(df):
    return np.exp(df)


def signed_power(df, pow):
    return np.sign(df) * df.abs().pow(pow)



def power(df, n):
    return df.pow(n)


abs_ = abs
def abs(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.abs()
    else:
        return abs_(obj)


min_ = min
def min(df1, df2, *args, key=None):
    if isinstance(df1, (pd.DataFrame, pd.Series)) and isinstance(df2, (pd.DataFrame, pd.Series)):
        nan_mask = None
        for df in [df1, df2]:
            if not isinstance(df, pd.DataFrame):
                continue
            if nan_mask is None:
                nan_mask = df.isnull()
            else:
                nan_mask = nan_mask | df1.isnull()
        res = ifelse(df1 <= df2, df1, df2)

        if nan_mask is not None:
            res = res.where(~nan_mask)
        return res
    else:
        if key:
            return min_(df1, df2, *args, key=key)
        else:
            return min_(df1, df2, *args)


max_ = max
def max(df1, df2, *args, key=None):
    if isinstance(df1, (pd.DataFrame, pd.Series)) and isinstance(df2, (pd.DataFrame, pd.Series)):
        return -min(-df1, -df2)
    else:
        if key:
            return max_(df1, df2, *args, key=key)
        else:
            return max_(df1, df2, *args, key=key)


if __name__ == '__main__':
    print(abs(-1))
