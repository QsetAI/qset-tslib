import pandas as pd


def quantile_mask(w, quantiles=None, labels=None):
    if quantiles is None:
        quantiles = [0., 0.33, 0.66]
    if labels is None:
        labels = list(range(len(quantiles)))
    if len(labels) != len(quantiles):
        raise Exception('quantiles and labels have different lengthes')

    mask = pd.DataFrame(labels[0], columns=w.columns, index=w.index)

    for i, q in enumerate(quantiles[1:]):
        quant = w.quantile(q, axis=1)
        mask[w.sub(quant, axis=0) >= 0] = labels[i + 1]
    mask = mask.where(~w.isnull())
    return mask


def quantile_masks(w, quantiles=None, labels=None):
    if quantiles is None:
        quantiles = [0., 0.33, 0.66]
    if labels is None:
        labels = list(range(len(quantiles)))
    if len(labels) != len(quantiles):
        raise Exception('quantiles and labels have different lengthes')

    mask = pd.DataFrame(labels[0], columns=w.columns, index=w.index)
    for i, q in enumerate(quantiles[1:]):
        quant = w.quantile(q, axis=1)
        mask[w.sub(quant, axis=0) >= 0] = labels[i + 1]
    mask = mask.where(~w.isnull())
    return mask