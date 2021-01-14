import numpy as np


def brinson(returns, mask, benchmark_weights, portfolio_weights):
    unique_marks = np.unique(mask)
    TAA = np.tile(np.nan, np.shape(returns))
    SS = np.tile(np.nan, np.shape(returns))
    Inter = np.tile(np.nan, np.shape(returns))
    DIF = np.tile(np.nan, np.shape(returns))
    for i in np.arange(len(unique_marks)):
        # TAA
        sector = unique_marks[i]
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
