import pandas as pd

from sklearn import manifold


def TSNE(returns, n_components=2, perplexity=0, random_state=0):
    # IQR filtering on returns
    q1 = returns.quantile(0.25, axis=0)
    q3 = returns.quantile(0.75, axis=0)
    filter_t = q3 + 1.5 * (q3 - q1)
    filter_t = filter_t.to_frame().T
    filter_t = pd.concat([filter_t] * returns.shape[0])
    filter_t.index = returns.index
    filter_b = q3 + 1.5 * (q3 - q1)
    filter_b = filter_b.to_frame().T
    filter_b = pd.concat([filter_b] * returns.shape[0])
    filter_b.index = returns.index
    returns.where((returns - filter_b) > 0)
    rets_cor = filter_b.where(returns - filter_b < 0, returns)
    rets_cor = filter_t.where(rets_cor - filter_t > 0, returns)

    C = manifold.TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    coords = C.fit_transform(rets_cor.values.T)
    coords = pd.DataFrame(coords, index=rets_cor.columns, columns=["x", "y"])
    return coords


# todo: test
