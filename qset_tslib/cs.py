import numpy as np
import pandas as pd
import qset_tslib as tslib
from qset_tslib.cython.neutralize import cs_neutralize


def cs_mean(df, as_series=False, *args, **kwargs):
    res = df.mean(axis=1, *args, **kwargs)
    if not as_series:
        res = tslib.make_like(df, res)
    return res


def cs_sum(df, as_series=False, skipna=True, *args, **kwargs):
    res = df.sum(axis=1, skipna=skipna, *args, **kwargs)
    if not as_series:
        res = tslib.make_like(df, res)
    return res


def cs_count(df, as_series=False, *args, **kwargs):
    res = df.count(axis=1, *args, **kwargs)
    if not as_series:
        res = tslib.make_like(df, res)
    return res


def cs_norm(df, factor=1.0, max_w=None, max_iter=10, tol=1e-2):
    df = df.divide(df.abs().sum(axis=1, min_count=1), axis=0).multiply(
        factor, axis="index"
    )

    if max_w is not None:
        for _ in range(max_iter):
            df = tslib.ifelse(abs(df) > max_w, tslib.sign(df) * max_w, df)

            if df.abs().max(axis=0).max() < max_w + tol:
                break
    return df


def cs_max(df, as_series=False, *args, **kwargs):
    res = df.max(axis=1, *args, **kwargs)
    if not as_series:
        res = tslib.make_like(df, res)
    return res


def cs_min(df, as_series=False, *args, **kwargs):
    res = df.min(axis=1, *args, **kwargs)
    if not as_series:
        res = tslib.make_like(df, res)
    return res


def cs_quantile(df, q=0.5, as_series=False, *args, **kwargs):
    res = df.quantile(q, axis=1, *args, **kwargs)
    if not as_series:
        res = tslib.make_like(df, res)
    return res


def cs_rank(data):
    return (data.rank(axis=1) - 1.0).div(data.count(axis=1) - 1.0, axis=0)


def cs_balance(alpha, max_w=None):

    alpha_positive = alpha[alpha > 0]
    alpha_negative = alpha[alpha < 0]

    res = tslib.make_like(alpha_positive)
    res[alpha > 0] = 0.5 * cs_norm(alpha_positive, max_w=max_w)
    res[alpha < 0] = 0.5 * cs_norm(alpha_negative, max_w=max_w)

    return res


def cs_normalize(df, max_w=None, tol=1e-2, max_iter=10):
    df = cs_norm(tslib.cs_neutralize(df))

    if max_w is not None:
        for _ in range(max_iter):
            df = tslib.ifelse(abs(df) > max_w, tslib.sign(df) * max_w, df)

            if df.abs().max(axis=0).max() < max_w + tol:
                break


def cs_threshold(df, value, threshold_type="rel", side="top", inclusive=False):
    if threshold_type == "abs":
        compare_df = df
    elif threshold_type == "rel":
        compare_df = cs_rank(df)
    else:
        raise Exception(f"Unknown threshold type: {threshold_type}")

    if side == "top":
        if inclusive:
            return compare_df >= value
        else:
            return compare_df > value
    elif side == "bottom":
        if inclusive:
            return compare_df <= value
        else:
            return compare_df < value
    else:
        raise Exception(f"Unknown side: {side}")


def test_cs():
    df = pd.DataFrame([np.arange(-1, 4, 1)] * 5)
    print(df)
    for key in [
        "mean",
        "max",
        "min",
        "sum",
        "count",
        "norm",
        "balance",
        "rank",
        "neutralize",
    ]:
        func = globals()[f"cs_{key}"]
        print(key)
        print(func(df))

    print(cs_threshold(df, 0.5))


if __name__ == "__main__":
    test_cs()
