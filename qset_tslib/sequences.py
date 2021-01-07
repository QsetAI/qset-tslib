import numpy as np


def duration(series):
    df = pd.DataFrame(series)
    df.columns = ['regime']
    df['regime'] = (df['regime'].diff() != 0 | df['regime'].isna()).astype(int).cumsum()
    df['regime'][series.isna()] = np.nan
    df['tmp'] = np.ones(len(series))
    grouped = df.groupby('regime').transform('cumsum')
    ret_series = grouped['tmp']
    ret_series.name = series.name
    return ret_series
