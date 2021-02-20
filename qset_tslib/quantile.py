import pandas as pd


def quantile_mask(df, quantiles=None, labels=None):
    quantiles = quantiles or [0.0, 0.33, 0.66]
    labels = labels or list(range(len(quantiles)))

    assert len(labels) == len(quantiles), "quantiles and labels have different length"

    mask = pd.DataFrame(labels[0], columns=df.columns, index=df.index)

    for i, q in enumerate(quantiles[1:]):
        quantile = df.quantile(q, axis=1)
        mask[df.sub(quantile, axis=0) >= 0] = labels[i + 1]
    mask = mask.where(~df.isnull())
    return mask
