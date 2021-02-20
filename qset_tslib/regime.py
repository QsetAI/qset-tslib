import numpy as np
import pandas as pd


def regime(s):
    df = pd.DataFrame(s, columns=["regime"])
    df["regime"] = (df["regime"].diff() != 0 | df["regime"].isna()).astype(int).cumsum()
    df["regime"][s.isna()] = np.nan
    return df["regime"]


def regime_duration(s):
    df = pd.DataFrame(regime(s))
    df["tmp"] = np.ones(len(s))
    grouped = df.groupby("regime").transform("cumsum")
    ret_series = grouped["tmp"]
    ret_series.name = s.name
    return ret_series


def test_regime():
    s = pd.Series([1, 1, 1, 2, np.nan, 2, 2])
    print(regime(s))
    print(regime_duration(s))


if __name__ == "__main__":
    test_regime()
