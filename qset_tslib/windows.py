import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def fixed_window_transform(df, window_size, transform_func='first'):
    df['tmp'] = np.arange(len(df)) // window_size
    return df.groupby('tmp').transform(transform_func)


def test_fixed_window_transform():
    df = pd.DataFrame(np.arange(100), columns=['foo'], index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)])
    print(df)
    print(fixed_window_transform(df, 20))


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

def rescale(df, freq_obj, func='first', closed='left', backfill=False):
    grouper = pd.Grouper(freq=cast_dateoffset(freq_obj), closed=closed)
    res = df.groupby(grouper).agg(func)

    if backfill:
        res = ts_backfill(res)

    return res

if __name__ == '__main__':
    test_fixed_window_transform()