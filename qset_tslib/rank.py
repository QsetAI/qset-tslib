

def rank(df, a=0., b=1., axis=1):
    """ A simplified version of pd.DataFrame.rank with specified delimiters. """
    return df.rank(axis=axis, pct=True) * (b - a) + a
