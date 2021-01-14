import pandas as pd
import numpy as np


def ts_ls_slope(df_x, df_y, window, min_periods=2, add_notna_mask=False, ddof=1):
    """ Calculates running slope of simple linear regression between windows of dataframes df_x and df_y, where df_y depends on df_x.
    """

    if add_notna_mask:
        _mask = df_x.notnull() & df_y.notnull()
        df_x = df_x.where(_mask)
        df_y = df_y.where(_mask)

    cov_xy = ((df_x * df_y).rolling(window, min_periods=min_periods).mean() -
              df_x.rolling(window, min_periods=min_periods).mean() * df_y.rolling(window, min_periods=min_periods).mean())
    var_x = df_x.rolling(window, min_periods=min_periods).var(ddof=ddof)
    return cov_xy / var_x


def ts_ls_slope_by_timeline(df, *args, **kwargs):
    """ Calculates rolling_ls_slope for every columns in dataframe as df_y and with range(len(df)) as df_x.
    """
    n_cols = len(df.columns)
    df_x = pd.DataFrame(np.repeat([np.arange(len(df))], n_cols, axis=0).T, index=df.index, columns=df.columns)
    df_y = df
    return ts_ls_slope(df_x, df_y, *args, **kwargs)
