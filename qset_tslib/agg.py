import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import qset_tslib as tslib
from qset_tslib.utils import cast_dateoffset


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


def agg_by_frequency(df, freq_obj, func="first", closed="left", backfill=False):
    grouper = pd.Grouper(freq=cast_dateoffset(freq_obj), closed=closed)
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
