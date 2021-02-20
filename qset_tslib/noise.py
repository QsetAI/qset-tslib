def filter_small_changes(df, eps, enhanced=True):
    changes = df.diff()
    large_changes = abs(changes) > eps
    res = df[large_changes].ffill()
    if enhanced:
        large_changes = large_changes | (abs(res - df) > eps)
        res = df[large_changes].ffill()
    return res
