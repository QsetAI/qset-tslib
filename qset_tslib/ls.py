import pandas as pd
import numpy as np

from qset_tslib.dataseries import mask


def ts_ls_slope(x, y, window, min_periods=2, add_notna_mask=False, ddof=1):
    """ Calculates running slope of simple linear regression between windows of numeric vectors x and y, where y depends on x.

    Parameters
    ----------
    x: pd.DataFrame or pd.Series
    y: pd.DataFrame or pd.Series
    window: int
        window size
    min_periods: int
        Minimum number of observations in window required to have a value (otherwise result is NA). For a window that is specified by an offset, this will default to 1.
    Returns
    -------
    pd.DataFrame
    """

    mp = min_periods
    if add_notna_mask:
        _mask = x.notnull() & y.notnull()
        _x = mask(x, _mask)
        _y = mask(y, _mask)
    else:
        _x = x
        _y = y

    cov_xy = ((_x * _y).rolling(window, min_periods=mp).mean() -
              _x.rolling(window, min_periods=mp).mean() * _y.rolling(window, min_periods=mp).mean())
    var_x = _x.rolling(window, min_periods=mp).var(ddof=ddof)
    return cov_xy / var_x


def ts_ls_slope_by_timeline(df, window=252, min_periods=2, add_notna_mask=False, ddof=1):
    """ Calculates rolling_ls_slope for every columns in dataframe as y and with range(len(df)) as x.

    Parameters
    ----------
    df: pd.DataFrame
    window: int
        window size
    min_periods: int
        Minimum number of observations in window required to have a value (otherwise result is NA). For a window that is specified by an offset, this will default to 1.
    Returns
    -------
    pd.DataFrame
    """
    n_cols = len(df.columns)
    x = pd.DataFrame(np.repeat([np.arange(len(df))], n_cols, axis=0).T, index=df.index, columns=df.columns)
    return ts_ls_slope(x, df, window=window, min_periods=min_periods, add_notna_mask=add_notna_mask, ddof=ddof)

