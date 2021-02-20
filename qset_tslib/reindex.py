import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import qset_tslib as tslib


def reindex_daily(df, keep="first"):
    """ Convert dataframe to daily index, leaving only one value for each date. """
    df["date"] = pd.Series(df.index, index=df.index).dt.date
    res = df.drop_duplicates(subset="date", keep=keep).set_index("date")
    res.index = pd.to_datetime(res.index)
    df.pop("date")
    return res


def test_reindex_daily():
    df = pd.DataFrame(
        np.arange(100),
        columns=["foo"],
        index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)],
    )
    print(df)
    print(reindex_daily(df, keep="first"))
    print(reindex_daily(df, keep="last"))


def reindex_at_days_of_month(df, days_of_month, keep="first"):
    df["date"] = pd.to_datetime(df.index.date)
    res = df.drop_duplicates(subset="date", keep=keep)
    res["day_of_month"] = res.date.dt.day
    res = res[res.day_of_month.isin(days_of_month)]
    res.drop(["date", "day_of_month"], inplace=True, axis=1)
    res = res.reindex(df.index)
    res = tslib.ts_backfill(res, len(df))
    df.pop("date")
    return res


def ts_reindex_at_weekdays(df, calc_weekdays, keep="first"):
    df["date"] = pd.to_datetime(df.index.date)
    res = df.drop_duplicates(subset="date", keep=keep)
    res.at[:, "weekday"] = res.date.dt.weekday
    res = res[res.weekday.isin(calc_weekdays)]
    res.drop(["date", "weekday"], inplace=True, axis=1)
    res = res.reindex(df.index)
    res = tslib.ts_backfill(res, len(df))
    df.pop("date")
    return res


def test():
    df = pd.DataFrame(
        np.arange(100),
        columns=["foo"],
        index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)],
    )
    print(reindex_daily(df))
    print(reindex_at_days_of_month(df, [1, 3]))
    print(ts_reindex_at_weekdays(df, [1, 3]))


if __name__ == "__main__":
    test()
