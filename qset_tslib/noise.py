
def ignore_small_changes(df, eps, enhance=True):
    changes = df.diff()
    big_changes = abs(changes) > eps
    res = df[big_changes].ffill()
    if enhance:
        big_changes = big_changes | (abs(res - df) > eps)
        res = df[big_changes].ffill()
    return res

