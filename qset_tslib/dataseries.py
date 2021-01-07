import pandas as pd
import numpy as np

import qset_tslib

from utils_ak.time import *


def mask(df, condition):
    return df.where(condition)


def rank(df, a=0., b=1., axis=1):
    """ A simplified version of pd.DataFrame.rank with specified delimiters. """
    return df.rank(axis=axis, pct=True) * (b - a) + a


def quantile_mask(w, quantiles=None, labels=None):
    if quantiles is None:
        quantiles = [0., 0.33, 0.66]
    if labels is None:
        labels = list(range(len(quantiles)))
    if len(labels) != len(quantiles):
        raise Exception('quantiles and labels have different lengthes')

    mask = pd.DataFrame(labels[0], columns=w.columns, index=w.index)

    for i, q in enumerate(quantiles[1:]):
        quant = w.quantile(q, axis=1)
        mask[w.sub(quant, axis=0) >= 0] = labels[i + 1]
    mask = mask.where(~w.isnull())
    return mask


def quantile_masks(w, quantiles=None, labels=None):
    if quantiles is None:
        quantiles = [0., 0.33, 0.66]
    if labels is None:
        labels = list(range(len(quantiles)))
    if len(labels) != len(quantiles):
        raise Exception('quantiles and labels have different lengthes')

    mask = pd.DataFrame(labels[0], columns=w.columns, index=w.index)
    for i, q in enumerate(quantiles[1:]):
        quant = w.quantile(q, axis=1)
        mask[w.sub(quant, axis=0) >= 0] = labels[i + 1]
    mask = mask.where(~w.isnull())
    return mask




def ts_recalc_at_days_of_month(df, days_of_month, keep='first'):
    """
    Take only first observations on particular days of month (1-31) and ts_backfill them
    :param df: DataFrame to apply transformation to
    :param days_of_month: list of integers between 1 and 31
    :param keep: keep from pd.drop_duplicates
    :return: DataFrame, with recalculated values for first observation on particular days of month, ts_backfilled otherwise

    example1 - leave values for 1, 8, 15, 22 month days, eg 1st October, 8th October, etc,  for each  month.
    dp = barsim.get_data_provider()
    cl = dp.get_data('close')
    periodic_cl = ts_recalc_at_days_of_month(cl, days_of_month = [1,8,15,22])

    example2 - leave only first value for each month, and fill nans with it
    dp = barsim.get_data_provider()
    cl = dp.get_data('close')
    periodic_cl = ts_recalc_at_days_of_month(cl, days_of_month = [1])
    """
    df['date'] = pd.to_datetime(df.index.date)
    res = df.drop_duplicates(subset='date', keep=keep)
    res['day_of_month'] = res.date.dt.day
    res = res[res.day_of_month.isin(days_of_month)]
    res.drop(['date', 'day_of_month'], inplace=True, axis=1)
    res = res.reindex(df.index)
    res = ts_backfill(res, len(df))
    df.pop('date')
    return res


def ts_recalc_at_weekdays(df, calc_weekdays, keep='first'):
    """
    Take only first observations on particular weekdays (0-6) and ts_backfill them
    :param df: DataFrame to transform
    :param calc_weekdays: list of integers between 0-6, 0 - Monday, 6-Sunday
    :param keep: keep from pd.drop_duplicates
    :return: DataFrame, with recalculated values for first observation on particular weekdays, ts_backfilled otherwise

    example1 - leave first value on Monday and Thursday, ts_backfill otherwise
    dp = barsim.get_data_provider()
    cl = dp.get_data('close')
    periodic_cl = ts_recalc_at_weekdays(cl, weekdays = [0,3])

    example2 - leave first value on Monday, ts_backfill otherwise
    dp = barsim.get_data_provider()
    cl = dp.get_data('close')
    periodic_cl = ts_recalc_at_weekdays(cl, weekdays = [0])
    """
    df['date'] = pd.to_datetime(df.index.date)
    res = df.drop_duplicates(subset='date', keep=keep)
    res['weekday'] = res.date.dt.weekday
    res = res[res.weekday.isin(calc_weekdays)]
    res.drop(['date', 'weekday'], inplace=True, axis=1)
    res = res.reindex(df.index)
    res = ts_backfill(res, len(df))
    df.pop('date')
    return res


def zscore(df, window, min_value=-3., max_value=3., min_periods=2):
    """ Zscore. """
    res = (df - ts_mean(df, window, min_periods=min_periods)) / ts_std(df, window, min_periods=min_periods)
    res = ifelse(res > max_value, max_value, res)
    res = ifelse(res < min_value, min_value, res)
    return res


# Basic ts functions


def duration(series):
    df = pd.DataFrame(series)
    df.columns = ['regime']
    df['regime'] = (df['regime'].diff() != 0 | df['regime'].isna()).astype(int).cumsum()
    df['regime'][series.isna()] = np.nan
    df['tmp'] = np.ones(len(series))
    grouped = df.groupby('regime').transform('cumsum')
    ret_series = grouped['tmp']
    ret_series.name = series.name
    return ret_series


def trunc_normalize(df, max_w, tol=5E-3, max_iter=10):
    """
    :param max_iter: max number of iterations
    :param tol: tolerance
    :param df: normalized df
    :param max_w: max allowed weight
    """
    df_out = normalize(df)

    n_iter = 1
    while df_out.abs().max(axis=0).max() > max_w + tol and n_iter <= max_iter:
        df_out = ifelse(abs(df_out) > max_w, barsim.math.sign(df_out) * max_w, df_out)
        df_out = normalize(df_out)
        n_iter += 1
    return df_out


def trunc_rescale(df, max_w, tol=5E-3, max_iter=10):
    """
    :param booksize: booksize
    :param max_iter: max number of iterations
    :param tol: tolerance
    :param df: df
    :param max_w: max allowed weight
    """

    df_out = norm(df)
    n_iter = 1
    while df_out.abs().max(axis=0).max() > max_w + tol and n_iter <= max_iter:
        df_out = ifelse(abs(df_out) > max_w, barsim.math.sign(df_out) * max_w, df_out)
        df_out = norm(df_out)
        n_iter += 1
    return df_out


trunc_norm = trunc_rescale


def fillna(df, *args, **kwargs):
    return df.fillna(*args, **kwargs)


# this is per-instrument
def rolling_rescale_1(df, interval=12 * 24):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :param interval:
    :return:
    """
    av_position = ts_mean(abs(df), interval)  # average position over interval
    norm_coeff = max(av_position, abs(df)).div(df.count(axis=1), axis=0)
    return df.div(norm_coeff, axis=0)


# this is altogether.
def rolling_rescale_2(df, interval=12 * 24):
    booksize = cs_sum(abs(df))  # booksize
    av_booksize = ts_mean(booksize, interval)  # average booksize over interval
    norm_coeff = max(av_booksize, booksize)
    return df.div(norm_coeff, axis=0)


def normalize(df):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: norm(neutralize(data))
    """
    return cs_norm(barsim.tools.neutralize(df))


# deprecated
# def decay(df, periods):
#     """
#     :param df:
#     :param periods:
#     :return: Linear combination of data with its previous values.
#      Slows down the signal. Reduces noise.
#     sum_(i=1)^periods ( w_i * lag(data, i))/(sum(w_i))
#     """
#     return ts_mean(df, weights=range(1, periods + 1))


# deprecated
# def exp_decay(df, u, periods=None, fillna=True):
#     # todo: use pandas solution (df.rolling(periods).ewm())
#     periods = periods or int(1. / u)
#     if fillna:
#         df = df.fillna(value=0)
#     res = df
#     for _ in range(periods):
#         res = res.shift() * (1 - u) + df * u
#     return res


def ignore_small_changes(df, eps, enhance=True):
    changes = df.diff()
    big_changes = abs(changes) > eps
    res = df[big_changes].ffill()
    if enhance:
        big_changes = big_changes | (abs(res - df) > eps)
        res = df[big_changes].ffill()
    return res


def combine(*dfs):
    """
    :param dfs:
    :return:
    """
    res = dfs[0].fillna(value=0)
    for df in dfs[1:]:
        res += df.fillna(value=0)
    return res


meta = combine


def valid(df):
    return np.sign(abs(df) + 1)


def fill_dataframe(df, series):
    df[:] = pd.concat([series] * df.shape[1], axis=1)
    return df


def comply_axes(target, source):
    return target.loc[source.index, source.columns]


def trunc_cs_balance(alpha, max_w):
    # todo: avoid upscale for days with values close to 0

    alpha_positive = alpha[alpha > 0]
    alpha_negative = alpha[alpha < 0]

    res = constant(np.nan, alpha_positive)
    res[alpha > 0] = 0.5 * trunc_norm(alpha_positive, max_w)
    res[alpha < 0] = 0.5 * trunc_norm(alpha_negative, max_w)

    return res



def filter_columns(df, columns, reverse=False, other=np.nan):
    if isinstance(df, pd.DataFrame):
        mask = constant(False, df)
        mask.loc[:, columns] = True
        if reverse:
            mask = ~mask
        return ifelse(mask, df, other)
    elif isinstance(df, pd.Series):
        result = pd.Series(index=df.index, data=np.nan)
        result[columns] = df[columns]
        return result


def filter_symbols(w, symbols, reverse=False, other=np.nan):
    return filter_columns(w, symbols, reverse, other)


def rescale(df, freq_obj, func='first', closed='left', backfill=False):
    grouper = pd.Grouper(freq=cast_dateoffset(freq_obj), closed=closed)
    res = df.groupby(grouper).agg(func)

    if backfill:
        res = ts_backfill(res)

    return res


def agg(df, by=None, func=None, level=None, **kwargs):
    return df.groupby(by=by, level=level, **kwargs).agg(func)


def threshold(df, threshold_type, value, side='top', inclusive=False):
    """
    :param df:
    :param threshold_type:
    :param value:
    :param side: 'top' or 'bottom'
    :return:
    """
    if threshold_type == 'abs':
        compare_df = df
    elif threshold_type == 'rel':
        compare_df = cs_rank(df)
    else:
        raise Exception(f'Unknown threshold type: {threshold_type}')

    if side == 'top':
        if inclusive:
            return compare_df >= value
        else:
            return compare_df > value
    elif side == 'bottom':
        if inclusive:
            return compare_df <= value
        else:
            return compare_df < value
    else:
        raise Exception(f'Unknown side: {side}')


# def threshold(df, threshold_type, value):
#     """
#     :param df:
#     :param threshold_type:
#     :param value:
#     :param side: 'top' or 'bottom'
#     :return:
#     """
#     if threshold_type == 'abs':
#         return df > value
#     elif threshold_type == 'rel':
#         return cs_rank(df) > value
#     else:
#         raise Exception('Unknown threshold type')
