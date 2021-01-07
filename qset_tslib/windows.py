import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def fixed_window_transform(df, window_size, transform_func='first'):
    df['tmp'] = np.arange(len(df)) // window_size
    return df.groupby('tmp').transform(transform_func)


def test_fixed_window_transform():
    df = pd.DataFrame(np.arange(100), columns=['foo'], index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)])
    print(df)
    print(fixed_window_transform(df, 20))


if __name__ == '__main__':
    test_fixed_window_transform()