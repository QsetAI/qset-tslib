import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import qset_tslib as tslib


def transform_by_fixed_window(df, window_size, transform_func="first"):
    df["tmp"] = np.arange(len(df)) // window_size
    return df.groupby("tmp").transform(transform_func)


def test_transform_by_fixed_window():
    df = pd.DataFrame(
        np.arange(100),
        columns=["foo"],
        index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)],
    )
    print(df)
    print(transform_by_fixed_window(df, 20))


# from python-utils-ak 0.1.9
def _cast_freq(td, keys=None):
    keys = keys or ["d", "h", "m", "s"]
    d = td.days
    h = td.seconds // 3600
    m = (td.seconds // 60) % 60
    s = td.seconds - (3600 * h + 60 * m)
    vals = [d, h, m, s]

    res = ""
    for i in range(4):
        if vals[i] != 0:
            res += f"{vals[i]}{keys[i]}"
    return res


def _cast_dateoffset(td_obj):
    # https://pandas.pydata.org/pandas-docs/stable/timeseries.html
    return _cast_freq(td_obj, keys=["D", "H", "T", "S"])


def agg_by_frequency(df, freq_obj, func="first", closed="left", backfill=False):
    grouper = pd.Grouper(freq=_cast_dateoffset(freq_obj), closed=closed)
    res = df.groupby(grouper).agg(func)

    if backfill:
        res = tslib.ts_backfill(res)

    return res


def test_agg_by_frequency():
    df = pd.DataFrame(
        np.arange(100),
        columns=["foo"],
        index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)],
    )
    print(agg_by_frequency(df, "1d", func="first"))


if __name__ == "__main__":
    test_transform_by_fixed_window()
    test_agg_by_frequency()
