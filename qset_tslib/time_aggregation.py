import pandas as pd
import numpy as np

from datetime import datetime, timedelta


def make_daily(df, keep='first'):
    """ Convert dataframe to daily index, leaving only one value for each date. """
    df['date'] = pd.Series(df.index, index=df.index).dt.date
    res = df.drop_duplicates(subset='date', keep=keep).set_index('date')
    res.index = pd.to_datetime(res.index)
    df.pop('date')
    return res


def test_make_daily():
    df = pd.DataFrame(np.arange(100), columns=['foo'], index=[datetime(2020, 1, 1) + i * timedelta(hours=4) for i in range(100)])
    print(df)
    print(make_daily(df, keep='first'))
    print(make_daily(df, keep='last'))


if __name__ == '__main__':
    test_make_daily()
