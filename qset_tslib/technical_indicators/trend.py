import pandas as pd
import numpy as np

import qset_tslib as tslib


def macd(close, n_fast=12, n_slow=26):
    """Moving Average Convergence Divergence (MACD)
    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.
    https://en.wikipedia.org/wiki/MACD
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    emafast = tslib.ts_exp_decay(close, n_fast, min_periods=n_fast)
    emaslow = tslib.ts_exp_decay(close, n_slow, min_periods=n_fast)
    macd = emafast - emaslow
    return macd


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Signal)
    Shows EMA of MACD.
    https://en.wikipedia.org/wiki/MACD
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.
    Returns:
        pandas.Series: New feature generated.
    """
    emafast = tslib.ts_exp_decay(close, n_fast, min_periods=n_fast)
    emaslow = tslib.ts_exp_decay(close, n_slow, min_periods=n_slow)
    macd = emafast - emaslow
    macd_signal = tslib.ts_exp_decay(macd, n_sign)
    macd_signal = macd_signal.replace([np.inf, -np.inf], np.nan)
    return macd_signal


def macd_diff(close, n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Diff)
    Shows the relationship between MACD and MACD Signal.
    https://en.wikipedia.org/wiki/MACD
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    emafast = tslib.ts_exp_decay(close, n_fast, min_periods=n_fast)
    emaslow = tslib.ts_exp_decay(close, n_slow, min_periods=n_slow)
    macd = emafast - emaslow
    macd_signal = tslib.ts_exp_decay(macd, n_sign)
    macd_diff = macd - macd_signal
    macd_diff = macd_diff.replace([np.inf, -np.inf], np.nan)
    return macd_diff


def vortex_indicator_pos(high, low, close, n=14):
    """Vortex Indicator (VI)
    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    tr = max(high, tslib.ts_lag(close, 1)) - min(low, tslib.ts_lag(close, 1))
    trn = tslib.ts_sum(tr, n)

    vmp = abs(high - tslib.ts_lag(low, 1))

    vip = tslib.ts_sum(vmp, n) / trn
    vip = vip.replace([np.inf, -np.inf], np.nan)
    return vip


def vortex_indicator_neg(high, low, close, n=14):
    """Vortex Indicator (VI)
    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    tr = max(high, tslib.ts_lag(close, 1)) - min(low, tslib.ts_lag(close, 1))
    trn = tslib.ts_sum(tr, n)

    vmm = abs(low - tslib.ts_lag(high, 1))

    vip = tslib.ts_sum(vmm, n) / trn
    vip = vip.replace([np.inf, -np.inf], np.nan)
    return vip


def trix(close, n=15):
    """Trix (TRIX)
    Shows the percent rate of change of a triple exponentially smoothed moving
    average.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    ema1 = tslib.ts_exp_decay(close, n, min_periods=n)
    ema2 = tslib.ts_exp_decay(ema1, n, min_periods=n)
    ema3 = tslib.ts_exp_decay(ema2, n, min_periods=n)
    trix = tslib.ts_returns(ema3)
    trix *= 100
    trix = trix.replace([np.inf, -np.inf], np.nan)
    return trix


def mass_index(high, low, n=9, n2=25):
    """Mass Index (MI)
    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of the
    current trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n(int): n low period.
        n2(int): n high period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    amplitude = high - low
    ema1 = tslib.ts_exp_decay(amplitude, n, min_periods=n)
    ema2 = tslib.ts_exp_decay(ema1, n, min_periods=n)
    mass = ema1 / ema2
    mass = tslib.ts_sum(mass, n2)
    mass = mass.replace([np.inf, -np.inf], np.nan)
    return mass


def cci(high, low, close, n=20, c=0.015):
    """Commodity Channel Index (CCI)
    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        c(int): constant.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    pp = (high + low + close) / 3.0
    cci = (pp - tslib.ts_mean(pp, n)) / (c * tslib.ts_std(pp, n))
    cci = cci.replace([np.inf, -np.inf], np.nan)
    return cci


def dpo(close, n=20):
    """Detrended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    dpo = tslib.ts_lag(close, int((0.5 * n) + 1)) - tslib.ts_mean(close, n)
    dpo = dpo.replace([np.inf, -np.inf], np.nan)
    return dpo


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15):
    """KST Oscillator (KST)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    https://en.wikipedia.org/wiki/KST_oscillator
    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = tslib.ts_mean(tslib.ts_returns(close, r1), n1)
    rocma2 = tslib.ts_mean(tslib.ts_returns(close, r2), n2)
    rocma3 = tslib.ts_mean(tslib.ts_returns(close, r3), n3)
    rocma4 = tslib.ts_mean(tslib.ts_returns(close, r4), n4)
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst = kst.replace([np.inf, -np.inf], np.nan)
    return kst


def kst_sig(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9):
    """KST Oscillator (KST Signal)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst
    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        nsig(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = tslib.ts_mean(tslib.ts_returns(close, r1), n1)
    rocma2 = tslib.ts_mean(tslib.ts_returns(close, r2), n2)
    rocma3 = tslib.ts_mean(tslib.ts_returns(close, r3), n3)
    rocma4 = tslib.ts_mean(tslib.ts_returns(close, r4), n4)
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst_sig = tslib.ts_mean(kst, nsig)
    kst_sig = kst_sig.replace([np.inf, -np.inf], np.nan)
    return kst_sig


def ichimoku_a(high, low, n1=9, n2=26, visual=False):
    """Ichimoku Kinkō Hyō (Ichimoku)
    It identifies the trend and look for potential signals within that trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n1(int): n1 low period.
        n2(int): n2 medium period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    conv = 0.5 * tslib.ts_max(high, n1) + tslib.ts_min(low, n1)
    base = 0.5 * tslib.ts_max(high, n2) + tslib.ts_min(low, n2)

    spana = 0.5 * (conv + base)

    if visual:
        spana = tslib.ts_lag(spana, n2)

    spana = spana.replace([np.inf, -np.inf], np.nan)
    return spana


def aroon_up(close, n=25):
    """Aroon Indicator (AI)
    Identify when trends are likely to change direction (uptrend).
    Aroon Up - ((N - Days Since N-day High) / N) x 100
    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    aroon_up = (tslib.ts_argmax(close, n) + 1) / n * 100
    aroon_up = aroon_up.replace([np.inf, -np.inf], np.nan)
    return aroon_up


def aroon_down(close, n=25):
    """Aroon Indicator (AI)
    Identify when trends are likely to change direction (downtrend).
    Aroon Down - ((N - Days Since N-day Low) / N) x 100
    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    aroon_down = (tslib.ts_argmin(close, n) + 1) / n * 100
    aroon_down = aroon_down.replace([np.inf, -np.inf], np.nan).fillna(0)
    return aroon_down
