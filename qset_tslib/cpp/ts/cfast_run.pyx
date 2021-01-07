# distutils: language = c++
# distutils: sources = cpp.cpp

from libcpp.vector cimport vector
from libcpp cimport bool


"""
std::vector<double> run_mean(const std::vector<double>& v, const int window, const int min_valid = 1);
std::vector<double> run_var(const std::vector<double>& v, const int window, const int min_valid = 2, const bool mean_zero = false, const int extra_df = 0);
std::vector<double> run_sd(const std::vector<double>& v, const int window, const int min_valid = 2, const bool mean_zero = false, const int extra_df = 0) ;
std::vector<double> run_min(const std::vector<double>& v, const int window, const int min_valid = 1);
std::vector<double> run_max(const std::vector<double>& v, const int window, const int min_valid = 1);
std::vector<double> run_argmin(const std::vector<double>& v, const int window, const int min_valid = 1);
std::vector<double> run_argmax(const std::vector<double>& v, const int window, const int min_valid = 1) ;
std::vector<double> run_quantile(const std::vector<double> v, const int window, const double q, const int min_valid = 1) ;
std::vector<double> run_median(const std::vector<double> v, const int window, const int min_valid = 1) ;
std::vector<double> run_ls_slope(const std::vector<double> x, const std::vector<double> y, const int window, const int min_valid = 2);
std::vector<double> run_cov(const std::vector<double> x, const std::vector<double> y, const int window, const int min_valid = 2, const bool mean_zero_x = false, const bool mean_zero_y = false, const int extra_df = 0);
std::vector<double> run_cor(const std::vector<double> x, const std::vector<double> y, const int window, const int min_valid = 3, const bool mean_zero_x = false, const bool mean_zero_y = false);
std::vector<double> run_skew(const std::vector<double> x, const int window, const int min_valid = 2, const bool mean_zero = false) ;
std::vector<double> run_smooth(const std::vector<double> x, const std::vector<double> kernel, const int min_valid = 1, const bool keep_na = false);
std::vector<double> lag_vector(const std::vector<double>& x, const int lag = 1) ;
std::vector<double> run_mdd(const std::vector<double>& x, const int window, const int min_valid = 1);
std::vector<double> run_rank(const std::vector<double>& v, const int window, const double min_value = -1.0, const double max_value = 1.0, const int min_valid = 1);
std::vector<double> run_zscore(const std::vector<double>& v, const int window, const double min_value = -3.0, const double max_value = 3.0, const int min_valid = 2);
std::vector<double> run_tapply_mean(const std::vector<double>& v, const std::vector<int>& group, const int window, const int min_valid = 1);
"""

cdef extern from "cpp.h":
    cdef vector[double] crun_mean "run_mean"(vector[double], int, int);
    cdef vector[double] crun_var "run_var"(vector[double], int, int, bool, int);
    cdef vector[double] crun_sd "run_sd"(vector[double], int, int, bool, int);
    cdef vector[double] crun_rank "run_rank"(vector[double], int, double, double, int);
    cdef vector[double] crun_argmin "run_argmin"(vector[double], int, int);
    cdef vector[double] crun_decay "run_decay"(vector[double], vector[double], int, bool);

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