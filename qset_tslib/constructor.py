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


if __name__ == '__main__':
    test_fill_with_value()
