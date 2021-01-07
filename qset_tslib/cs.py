
def cs_mean(df, as_series=False):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: cross-sectional mean
    """
    res = df.mean(axis=1)
    if not as_series:
        res = fill_dataframe(df.copy_path(), res)
    return res


def cs_sum(df, as_series=False, skipna=True):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: cross-sectional mean
    """
    res = df.sum(axis=1, skipna=skipna)
    if not as_series:
        res = fill_dataframe(df.copy_path(), res)
    return res


def cs_count(df, as_series=False):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: number of valid (not nan) datapoints in each moment of time
    """
    res = df.count(axis=1)
    if not as_series:
        res = fill_dataframe(df.copy_path(), res)
    return res


def cs_norm(df, booksize=1.):
    """ Norm dataframe. """
    return df.divide(df.abs().sum(axis=1, min_count=1), axis=0).multiply(booksize, axis='index')


norm = cs_norm

def cs_max(df, as_series=False):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: cross-sectional maximum
    """
    res = df.max(axis=1)
    if not as_series:
        res = fill_dataframe(df.copy_path(), res)
    return res


def cs_min(df, as_series=False):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: cross-sectional minimum
    """
    res = df.min(axis=1)
    if not as_series:
        res = fill_dataframe(df.copy_path(), res)
    return res


def cs_quantile(df, q=0.5, as_series=False):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: Rolling cross-sectional quantile, linear approximation
    """
    res = df.quantile(q, axis=1)
    if not as_series:
        res = fill_dataframe(df.copy_path(), res)
    return res


def cs_rank(data):
    """
    :param data: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return:
    """
    return (data.rank(axis=1) - 1.).div(data.count(axis=1) - 1., axis=0)


def cs_balance(alpha):
    # todo: avoid upscale for days with values close to 0

    alpha_positive = alpha[alpha > 0]
    alpha_negative = alpha[alpha < 0]

    res = constant(np.nan, alpha_positive)
    res[alpha > 0] = 0.5 * norm(alpha_positive)
    res[alpha < 0] = 0.5 * norm(alpha_negative)

    return res
