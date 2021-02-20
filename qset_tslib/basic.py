import pandas as pd
import numpy as np


def ifelse(cond, df, other=np.nan, **kwargs):
    return df.where(cond, other=other, **kwargs)


def test_ifelse():
    df1 = pd.DataFrame(np.arange(0, 5, 1))
    df2 = df1.sort_values(by=0, ascending=False).reset_index(drop=True)  # reversed df
    print(ifelse(df1 > df2, df1, df2))
    print(ifelse(df1 < df2, df1, df2))


def make_like(df, fill_value=np.nan):
    df = df.copy()
    if isinstance(fill_value, pd.Series):
        df[:] = pd.concat([fill_value] * df.shape[1], axis=1)
    else:
        df[:] = fill_value
    return df


def test_make_like():
    df = pd.DataFrame([np.arange(0, 5, 1)] * 5)

    s = pd.Series(np.arange(10, 15, 1))
    c = 3

    print(df)
    print(make_like(df))
    print(make_like(df, s))
    print(make_like(df, c))


def comply_axes(target, source):
    return target.loc[source.index, source.columns]


def filter_columns(df, columns, reverse=False, other=np.nan):
    mask = make_like(df, False)
    mask.loc[:, columns] = True
    if reverse:
        mask = ~mask
    return ifelse(mask, df, other)


def test_filter_columns():
    df = pd.DataFrame([np.arange(0, 5, 1)] * 5)
    s = pd.Series(np.arange(10, 15, 1))

    print(filter_columns(df, [2, 3]))


if __name__ == "__main__":
    test_ifelse()
    test_make_like()
    test_filter_columns()
