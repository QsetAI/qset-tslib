from qset_tslib.cpp.ts.cts import *
import pandas as pd
import numpy as np


def test_ts():
    arr = [1., 2., 3., 4., 3., 2., 1.]
    print(run_mean(arr, window=3))
    print(run_var(arr, window=3))
    print(run_sd(arr, window=3))
    print(run_min(arr, window=3))
    print(run_max(arr, window=3))


if __name__ == '__main__':
    test_ts()

