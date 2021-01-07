
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



def normalize(df):
    """
    :param df: data, pandas.DataFrame(data loaded with data manager/obtained with opertaions)
    :return: norm(neutralize(data))
    """
    return cs_norm(barsim.tools.neutralize(df))


def trunc_normalize(df, max_w, tol=5E-3, max_iter=10):
    """
    :param max_iter: max number of iterations
    :param tol: tolerance
    :param df: normalized df
    :param max_w: max allowed weight
    """
    df_out = normalize(df)

    n_iter = 1
    while df_out.abs().max(axis=0).max() > max_w + tol and n_iter <= max_iter:
        df_out = ifelse(abs(df_out) > max_w, qset_tslib.math.sign(df_out) * max_w, df_out)
        df_out = normalize(df_out)
        n_iter += 1
    return df_out

def trunc_cs_balance(alpha, max_w):
    # todo: avoid upscale for days with values close to 0

    alpha_positive = alpha[alpha > 0]
    alpha_negative = alpha[alpha < 0]

    res = constant(np.nan, alpha_positive)
    res[alpha > 0] = 0.5 * trunc_norm(alpha_positive, max_w)
    res[alpha < 0] = 0.5 * trunc_norm(alpha_negative, max_w)

    return res

def threshold(df, threshold_type, value, side='top', inclusive=False):
    """
    :param df:
    :param threshold_type:
    :param value:
    :param side: 'top' or 'bottom'
    :return:
    """
    if threshold_type == 'abs':
        compare_df = df
    elif threshold_type == 'rel':
        compare_df = cs_rank(df)
    else:
        raise Exception(f'Unknown threshold type: {threshold_type}')

    if side == 'top':
        if inclusive:
            return compare_df >= value
        else:
            return compare_df > value
    elif side == 'bottom':
        if inclusive:
            return compare_df <= value
        else:
            return compare_df < value
    else:
        raise Exception(f'Unknown side: {side}')



def trunc_rescale(df, max_w, tol=5E-3, max_iter=10):
    """
    :param booksize: booksize
    :param max_iter: max number of iterations
    :param tol: tolerance
    :param df: df
    :param max_w: max allowed weight
    """

    df_out = norm(df)
    n_iter = 1
    while df_out.abs().max(axis=0).max() > max_w + tol and n_iter <= max_iter:
        df_out = ifelse(abs(df_out) > max_w, barsim.math.sign(df_out) * max_w, df_out)
        df_out = norm(df_out)
        n_iter += 1
    return df_out


trunc_norm = trunc_rescale


# def threshold(df, threshold_type, value):
#     """
#     :param df:
#     :param threshold_type:
#     :param value:
#     :param side: 'top' or 'bottom'
#     :return:
#     """
#     if threshold_type == 'abs':
#         return df > value
#     elif threshold_type == 'rel':
#         return cs_rank(df) > value
#     else:
#         raise Exception('Unknown threshold type')
