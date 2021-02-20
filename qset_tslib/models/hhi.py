import numpy as np
import pandas as pd


def HHI(returns, lookback=252 * 1, freq=21):
    hhi_index = []
    hhi_values = []
    for i in np.arange(lookback, returns.shape[0], freq):
        hhi_index.append(returns.index[i])
        cur_rets = returns.iloc[i - lookback : i]

        total_rets = cur_rets.sum(axis=1) / cur_rets.shape[1]
        U, s, V = np.linalg.svd(cur_rets, full_matrices=True)
        new_rets = np.dot(cur_rets, V.T)
        new_rets = pd.DataFrame(new_rets)

        c = []
        for j in range(returns.shape[1]):
            a = np.cov(total_rets, new_rets.values[:, j])[0, 0]
            b = np.cov(new_rets.values[:, j], new_rets.values[:, j])[0, 0]
            c.append(a / b if abs(b) > 1e-9 else 0)
        c = pd.DataFrame(c)
        c = c / c.abs().sum(axis=0)
        c = c ** 2
        hhi_values.append(1 / c.sum())
    return pd.DataFrame(hhi_values, index=hhi_index)


# todo: test
