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



if __name__ == '__main__':
    print(abs(-1))
