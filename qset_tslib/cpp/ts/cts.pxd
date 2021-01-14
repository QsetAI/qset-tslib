from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "ts.h":
    vector[double] run_mean(vector[double], int, int);
    vector[double] run_var(vector[double], int, int, bool, int);
    vector[double] run_sd(vector[double], int, int, bool, int);
    vector[double] run_min(vector[double], int, int);
    vector[double] run_max(vector[double], int, int);

    # vector[double] run_rank(vector[double], int, double, double, int);
    # vector[double] run_argmin(vector[double], int, int);
    # vector[double] run_decay(vector[double], vector[double], int, bool);