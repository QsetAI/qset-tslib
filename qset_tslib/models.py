import numpy as np
import pandas as pd

from sklearn import manifold


def HHI(rets, lookback=252 * 1, freq=21):
    hhi_index = []
    hhi_values = []
    for i in np.arange(lookback, rets.shape[0], freq):
        hhi_index.append(rets.index[i])
        cur_rets = rets.iloc[i - lookback:i]

        total_rets = cur_rets.sum(axis=1) / cur_rets.shape[1]
        U, s, V = np.linalg.svd(cur_rets, full_matrices=True)
        new_rets = np.dot(cur_rets, V.T)
        new_rets = pd.DataFrame(new_rets)

        c = []
        for j in range(rets.shape[1]):
            a = np.cov(total_rets, new_rets.values[:, j])[0, 0]
            b = np.cov(new_rets.values[:, j], new_rets.values[:, j])[0, 0]
            c.append(a / b if abs(b) > 1e-9 else 0)
        c = pd.DataFrame(c)
        c = c / c.abs().sum(axis=0)
        c = c ** 2
        hhi_values.append(1 / c.sum())
    return pd.DataFrame(hhi_values, index=hhi_index)


def TSNE(rets, n_components=2, perplexity=0, random_state=0):
    # IQR filtering on returns
    q1 = rets.quantile(0.25, axis=0)
    q3 = rets.quantile(0.75, axis=0)
    filter_t = q3 + 1.5 * (q3 - q1)
    filter_t = filter_t.to_frame().T
    filter_t = pd.concat([filter_t] * rets.shape[0])
    filter_t.index = rets.index
    filter_b = q3 + 1.5 * (q3 - q1)
    filter_b = filter_b.to_frame().T
    filter_b = pd.concat([filter_b] * rets.shape[0])
    filter_b.index = rets.index
    rets.where((rets - filter_b) > 0)
    rets_cor = filter_b.where(rets - filter_b < 0, rets)
    rets_cor = filter_t.where(rets_cor - filter_t > 0, rets)

    C = manifold.TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    coords = C.fit_transform(rets_cor.values.T)
    coords = pd.DataFrame(coords, index=rets_cor.columns, columns=['x', 'y'])
    return coords


def brinson(returns, mask, benchmark_weights, portfolio_weights):
    unique_marks = np.unique(mask)
    TAA = np.tile(np.nan, np.shape(returns))
    SS = np.tile(np.nan, np.shape(returns))
    Inter = np.tile(np.nan, np.shape(returns))
    DIF = np.tile(np.nan, np.shape(returns))
    for i in np.arange(len(unique_marks)):
        # TAA
        sector = unique_marks[i]
        print('current sector: ', sector)
        w_P = portfolio_weights.where(mask == sector)
        w_B = benchmark_weights.where(mask == sector)
        sector_returns = returns.where(mask == sector)
        r_B = (sector_returns * w_B).sum(axis=1)
        aa = (w_P - w_B).sum(axis=1) * r_B
        TAA[:, i] = aa
        # SS
        r_P = (sector_returns * w_P).sum(axis=1)
        ss = (w_B).sum(axis=1) * (r_P - r_B)
        SS[:, i] = ss
        # Interaction term
        it = (w_P - w_B).sum(axis=1) * (r_P - r_B)
        Inter[:, i] = it
        # diff
        diff = w_P.sum(axis=1) * r_P - w_B.sum(axis=1) * r_B
        DIF[:, i] = diff
        # break
    TAA = np.nansum(TAA, axis=1)
    SS = np.nansum(SS, axis=1)
    Inter = np.nansum(Inter, axis=1)
    DIF = np.nansum(DIF, axis=1)
    # TAA - returns because of sector selection
    # SS - stock selection
    # Inter - misc intercations
    # DIF - total value added
    return TAA, SS, Inter, DIF
