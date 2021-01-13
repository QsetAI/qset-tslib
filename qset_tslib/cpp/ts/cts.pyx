# distutils: language = c++
# distutils: sources = ts.cpp

from cts cimport run_mean as crun_mean
from cts cimport run_var as crun_var
from cts cimport run_sd as crun_sd
from cts cimport run_rank as crun_rank
from cts cimport run_argmin as crun_argmin
from cts cimport run_decay as crun_decay

def run_mean(arr, window, min_periods=1):
    return crun_mean(arr, window, min_periods)

def run_var(arr, window, min_periods=2, mean_zero=False, extra_df=0):
    return crun_var(arr, window, min_periods, mean_zero, extra_df)

def run_sd(arr, window, min_periods=2, mean_zero=False, extra_df=0):
    return crun_sd(arr, window, min_periods, mean_zero, extra_df)

def run_rank(arr, window, min_value, max_value, min_periods=1):
    return crun_rank(arr, window, min_value, max_value, min_periods)

def run_argmin(arr, window, min_periods=1):
    return crun_argmin(arr, window, min_periods)

def run_decay(arr, w, min_periods=1, exclude_nans=False):
    return crun_decay(arr, w, min_periods, exclude_nans)