import pandas as pd

try:
    from qset_tslib.cython.neutralize.cneutralize import neutralize as _neutralize
except ImportError:
    pass


def cs_neutralize(df, group=None, norm_std=False):
    if group is None:
        df = df.sub(df.mean(axis=1), axis=0)
        if norm_std:
            df = df.divide(df.std(axis=1), axis=0)
        return df
    return pd.DataFrame(
        _neutralize(df.values, group.values, norm_std=norm_std),
        index=df.index,
        columns=df.columns,
    )
