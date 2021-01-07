from barsim.tools import cfast_run
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd


def split_workload(df, num):
    cols = df.columns
    k, m = divmod(len(cols), num)
    return [cols[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num)]


def apply_to_dataframe(wrap_func, data, num_processes=None, **kwargs):
    ''' Apply a function separately to each column in a dataframe, in parallel.'''
    if isinstance(data, pd.DataFrame):

        num_processes = num_processes or min(data.shape[1], cpu_count()-1)
        with Pool(num_processes) as pool:
            parts = split_workload(data, num_processes)
            seq = [data[part] for part in parts]
            results_list = pool.map(partial(wrap_func, **kwargs), seq)
            return pd.concat(results_list, axis=1)
    elif isinstance(data, pd.Series):
        return wrap_func(data, **kwargs)
    else:
        raise Exception("only pd.Series and pd.Dataframe supported in apply_to_dataframe")


def wrap_ts_decay(data, **kwargs):
    if isinstance(data, pd.DataFrame):
        return data.apply(cfast_run.run_decay, **kwargs)
    elif isinstance(data, pd.Series):
        return pd.Series(data=cfast_run.run_decay(data, **kwargs), index=data.index)
    else:
        raise Exception("only pd.Series and pd.Dataframe supported in ts_decay")


def wrap_ts_rank(data, **kwargs):
    if isinstance(data, pd.DataFrame):
        return data.apply(cfast_run.run_rank, **kwargs)
    elif isinstance(data, pd.Series):
        return pd.Series(data=cfast_run.run_rank(data, **kwargs), index=data.index)
    else:
        raise Exception("only pd.Series and pd.Dataframe supported in ts_rank")


def ts_decay(df, window=None, w=None, exclude_nans=True, min_periods=1, ewa=False):
    """
    :param w: custom weights, overrides all
    :param exclude_nans: exclude NaN observations from mean calculations (zero weights in (de)nominator)
    :param ewa: use exp weighted average
    Note: to reproduce the behavior of Pandas' EWMA use ts_exp_decay() from dataseries.py
    PS: ts_exp_decay() sometimes results in lower turnover
    """
    if w is not None:
        _w = w
    elif window:
        if ewa:
            alpha = 2.0 / (1.0 + window)
            _w = [(1 - alpha) ** i for i in reversed(range(window))]
        else:
            _w = [i + 1 for i in range(window)]
    else:
        raise Exception('Incorrect arguments for ts_decay')
    return apply_to_dataframe(wrap_ts_decay, df, w=_w, exclude_nans=exclude_nans, min_periods=min_periods)


# deprecated
# def ts_decay_old(df, window=None, w=None, exclude_nans=True, min_periods=1, ewa=False):
#     """
#     :param w: custom weights, overrides all
#     :param exclude_nans: exclude NaN observations from mean calculations (zero weights in (de)nominator)
#     :param ewa: use exp weighted average
#     Note: to reproduce the behavior of Pandas' EWMA use ts_exp_decay() from dataseries.py
#     PS: ts_exp_decay() sometimes results in lower turnover
#     """
#     if w is not None:
#         _w = w
#     elif window:
#         if ewa:
#             alpha = 2.0 / (1.0 + window)
#             _w = [(1 - alpha) ** i for i in reversed(range(window))]
#         else:
#             _w = [i + 1 for i in range(window)]
#     else:
#         raise Exception('Incorrect arguments for ts_decay')
#     return df.apply(cfast_run.run_decay, w=_w, exclude_nans=exclude_nans, min_periods=min_periods)


def ts_rank(df, window, min_value=-1., max_value=1., min_periods=1, axis=0):
    return apply_to_dataframe(wrap_ts_rank, df, axis=axis, window=window, min_value=min_value, max_value=max_value,
                              min_periods=min_periods)


# deprecated
# def ts_rank_old(df, window, min_value=-1., max_value=1., min_periods=1, axis=0):
#     return df.apply(cfast_run.run_rank, axis=axis, window=window, min_value=min_value, max_value=max_value, min_periods=min_periods)


def ts_cmean(df, window, min_periods=1):
    return df.apply(cfast_run.run_mean, window=window, min_periods=min_periods)


def ts_argmin(df, window, min_periods=1):
    return df.apply(cfast_run.run_argmin, window=window, min_periods=min_periods)


def ts_argmax(df, window, min_periods=1):
    df = -df
    return df.apply(cfast_run.run_argmin, window=window, min_periods=min_periods)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    t = pd.DataFrame(pd.Series([16, np.nan, np.nan, 2, 77, np.nan, np.nan, np.nan, np.nan, np.nan, 29, 4, 5, 8, 2]))
    print(pd.concat([t, ts_decay(t, 3, True), ts_decay(t, 3, False)], axis=1))