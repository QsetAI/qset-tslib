import qset_tslib as tslib
import pandas as pd
import numpy as np


def max_dd(s):
    """
    :param s: pandas.Series - actually window
    :return: Returns drawdown for a window
    Meaning: subsidiary function for ts_drawdown
    """
    cumulative = s.cumsum()
    max2here = cumulative.cummax()
    dd2here = cumulative - max2here
    return dd2here.min()


def ts_drawdown(df, window=1, min_periods=0):
    """
      :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
      :param ndays: number of days
      :param min_periods: minimal non-NaNs values in window to make a calculation.
      The bigger min_periods is, the later is first non-NaN value calculated.
      :return: rolling absolute drawdown. Values are zero or negative.
      Example: for data
      [1,2,5,4,3,2,3], ts_drawdown(data, ndays=3, min_periods=2) returns
    [NaN,0,0,1,2,2,0]
    """
    return df.rolling(window=window, min_periods=min_periods).apply(max_dd)


def ts_drawup(data, ndays=1, min_periods=0):
    """
    :param data: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param ndays: number of days
    :param min_periods: minimal non-NaNs values in window to make a calculation.
    The bigger min_periods is, the later is first non-NaN value calculated.
    :return: rolling absolute drawup. See ts_drawdown. Values are zero or positive.
    """
    return -ts_drawdown(-data, ndays, min_periods=min_periods)


def ts_returns(df, periods=1, upper_bound=None, lower_bound=None, fill_method="pad"):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of periods
    :param upper_bound: if returns value is greater than upper bound replace it with upper bound. - outlier control
    :param lower_bound: if returns value is less than lower bound replace it with lower bound. - outlier control
    :return: Returns of data over n days. (data_di - data_(di-periods))/(data_(di-periods))
    Meaning: how much 1 dollar invested in the asset n days ago would have earned
    values capped at -0.8 and 4
    """
    res = df.pct_change(periods, fill_method=fill_method)
    if upper_bound is not None:
        res[res > upper_bound] = upper_bound
    if lower_bound is not None:
        res[res < lower_bound] = lower_bound
    return res


def ts_mean(df, periods=None, weights=None, norm=True, min_periods=None, win_type=None):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Rolling average over n days if weights is None.
             Otherwise, the resulting formula is (df.shift(0) * w[-1] + df.shift(1) * w[-2] + ... + df.shift(len(w) - 1) * w[0]) / np.abs(w).sum()

    Notes
    -----
    Working with nans is not handled properly. For proper nan handling, a c++ implementation is needed
    """
    if not periods and not weights:
        raise Exception("periods or weights should be specified")

    if weights is None:
        return df.rolling(periods, min_periods=min_periods, win_type=win_type).mean()

    res = None
    w_res = None
    ones = pd.DataFrame(1.0, index=df.index, columns=df.columns)
    for i in range(len(weights)):
        w = weights[-(i + 1)]
        dfi = df.shift(i)
        if res is None:
            res = w * dfi.fillna(0)
            w_res = abs(w) * ones.where(~dfi.isnull(), 0.0)
        else:
            res += w * dfi.fillna(0)
            w_res += abs(w) * ones.where(~dfi.isnull(), 0.0)
    if norm:
        res /= w_res
    return res


def ts_exp_decay(df, window, min_periods=1, ignore_na=False, adjust=True):
    return df.ewm(
        span=window, min_periods=min_periods, ignore_na=ignore_na, adjust=adjust
    ).mean()


def ts_vwap(close, volume, periods):
    return ts_sum((close * volume), periods) / ts_sum(volume, periods)


def ts_min(df, periods, min_periods=None):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Rolling minimum over n days
    """
    return df.rolling(periods, min_periods=min_periods).min()


def ts_max(df, periods, min_periods=None):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Rolling maximum over n days
    """
    return df.rolling(periods, min_periods=min_periods).max()


def ts_sum(df, periods, min_periods=1):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Rolling sum over n days
    """
    return df.rolling(periods, min_periods=min_periods).sum()


def ts_lag(df, periods=1):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Value of data n days ago
    """
    return df.shift(periods)


def ts_diff(df, periods=1):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Difference of data value today and n days ago
    """
    return df.diff(periods)


def ts_median(df, periods, min_periods=1):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Rolling median over n days
    """
    return df.rolling(periods, min_periods=min_periods).median()


def ts_quantile(df, periods, min_periods=1, q=0.5):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods: number of days
    :return: Rolling quantile over n days
    """
    return df.rolling(periods, min_periods=min_periods).quantile(q)


# todo: rename
def ts_backfill(df, limit=None):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param periods:
    :return: fill missing datapoints with previous data.
    """
    return df.fillna(method="ffill", limit=limit)


def ts_regime(df):
    return df.apply(tslib.regime)


def ts_regime_duration(df):
    return df.apply(tslib.regime_duration)


def ts_std(df, periods, min_periods=1):
    # return (data.rank(axis=1)
    return df.rolling(periods, min_periods=min_periods).std()


def ts_parkinson_vol(high, low, periods):
    return tslib.log(high / low).rolling(periods).mean() * np.sqrt(np.pi / 8)


def ts_corr(
    df1, df2=None, periods=None, min_periods=None, pairwise=False, nan_incorrect=True
):
    corr_df = df1.rolling(window=periods, min_periods=min_periods).corr(
        df2, pairwise=pairwise
    )
    if nan_incorrect:
        corr_df = tslib.ifelse(
            corr_df.isin([np.inf, -np.inf]), tslib.make_like(corr_df, np.nan), corr_df
        )
        tol = 1e-5
        corr_df = tslib.ifelse(
            abs(corr_df) > 1 + tol, tslib.make_like(corr_df, np.nan), corr_df
        )
    return corr_df


def ts_zscore(df, window, min_value=-3.0, max_value=3.0, min_periods=2):
    """ Zscore. """
    res = (df - ts_mean(df, window, min_periods=min_periods)) / ts_std(
        df, window, min_periods=min_periods
    )
    res = tslib.ifelse(res > max_value, max_value, res)
    res = tslib.ifelse(res < min_value, min_value, res)
    return res


def ts_rescale(df, window=12 * 24):
    average_position = ts_mean(abs(df), window)
    norm_coeff = max(average_position, abs(df)).div(df.count(axis=1), axis=0)
    return df.div(norm_coeff, axis=0)


def tcs_rescale(df, window=12 * 24):
    booksize = tslib.cs_sum(abs(df))  # booksize
    av_booksize = ts_mean(booksize, window)  # average booksize over interval
    norm_coeff = max(av_booksize, booksize)
    return df.div(norm_coeff, axis=0)
