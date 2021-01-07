import pandas as pd
import numpy as np


def fill_with_value(df, fill_value):
    df = df.copy()
    if isinstance(fill_value, pd.Series):
        df[:] = pd.concat([fill_value] * df.shape[1], axis=1)
    else:
        df[:] = fill_value
    return df


def test_fill_with_value():
    df = pd.DataFrame([np.arange(0, 5, 1)] * 5)

    s = pd.Series(np.arange(10, 15, 1))
    c = 3

    print(df)
    print(fill_with_value(df, s))
    print(fill_with_value(df, c))


def comply_axes(target, source):
    return target.loc[source.index, source.columns]


def combine(*dfs):
    """
    :param dfs:
    :return:
    """
    res = dfs[0].fillna(value=0)
    for df in dfs[1:]:
        res += df.fillna(value=0)
    return res


meta = combine


def filter_columns(df, columns, reverse=False, other=np.nan):
    if isinstance(df, pd.DataFrame):
        mask = constant(False, df)
        mask.loc[:, columns] = True
        if reverse:
            mask = ~mask
        return ifelse(mask, df, other)
    elif isinstance(df, pd.Series):
        result = pd.Series(index=df.index, data=np.nan)
        result[columns] = df[columns]
        return result


def filter_symbols(w, symbols, reverse=False, other=np.nan):
    return filter_columns(w, symbols, reverse, other)


if __name__ == '__main__':
    test_fill_with_value()
