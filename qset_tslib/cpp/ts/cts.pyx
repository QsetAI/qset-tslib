# distutils: language = c++
# distutils: sources = ts.cpp

from cts cimport run_mean as _run_mean
from cts cimport run_var as _run_var
from cts cimport run_sd as _run_sd
from cts cimport run_min as _run_min
from cts cimport run_max as _run_max

# from cts cimport run_rank as _run_rank
# from cts cimport run_argmin as _run_argmin
# from cts cimport run_decay as _run_decay

def run_mean(arr, window, min_periods=1):
    return _run_mean(arr, window, min_periods)

def run_var(arr, window, min_periods=2, mean_zero=False, extra_df=0):
    return _run_var(arr, window, min_periods, mean_zero, extra_df)

def run_sd(arr, window, min_periods=2, mean_zero=False, extra_df=0):
    return _run_sd(arr, window, min_periods, mean_zero, extra_df)

def run_min(arr, window, min_periods=1):
    return _run_min(arr, window, min_periods)

def run_max(arr, window, min_periods=1):
    return _run_max(arr, window, min_periods)


# def run_rank(arr, window, min_value=-1.0, max_value=1.0, min_periods=1):
#     return _run_rank(arr, window, min_value, max_value, min_periods)

# def run_argmin(arr, window, min_periods=1):
#     return _run_argmin(arr, window, min_periods)
#
# def run_decay(arr, w, min_periods=1, exclude_nans=False):
#     return _run_decay(arr, w, min_periods, exclude_nans)