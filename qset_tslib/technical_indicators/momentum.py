import pandas as pd
import numpy as np

import qset_tslib as tslib


def money_flow_index(high, low, close, volume, n=14):

    up_or_down = tslib.ifelse(
        tslib.ts_diff(close) > 0, tslib.make_like(close, 1), tslib.make_like(close, -1)
    )

    # 1 typical price
    tp = (high + low + close) / 3.0

    # 2 money flow
    mf = tp * volume

    positive_mf = tslib.ifelse(up_or_down > 0, mf, tslib.make_like(mf, 0))
    negative_mf = tslib.ifelse(up_or_down < 0, mf, tslib.make_like(mf, 0))
    # 3 positive and negative money flow with n periods
    mfr = tslib.ts_sum(positive_mf, n) / tslib.ts_sum(negative_mf, n)
    mfi = 100 - 100 / (1 + mfr)
    mfi = mfi.replace([np.inf, -np.inf], np.nan)
    return mfi


def true_strength_index(close, r=25, s=13, fillna=False):
    """True strength index (TSI)
    Shows both trend direction and overbought/oversold conditions.
    https://en.wikipedia.org/wiki/True_strength_index
    Args:
        close(pandas.Series): dataset 'Close' column.
        r(int): high period.
        s(int): low period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    m = tslib.ts_diff(close)
    m1 = m.ewm(r).mean().ewm(s).mean()
    m2 = abs(m).ewm(r).mean().ewm(s).mean()
    tsi = m1 / m2
    tsi *= 100
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan)
    return tsi


def ultimate_oscillator(high, low, close, s=7, m=14, l=28, ws=4.0, wm=2.0, wl=1.0):
    """Ultimate Oscillator
    Larry Williams' (1976) signal, a momentum oscillator designed to capture momentum
    across three different timeframes.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)
    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        s(int): short period
        m(int): medium period
        l(int): long period
        ws(float): weight of short BP average for UO
        wm(float): weight of medium BP average for UO
        wl(float): weight of long BP average for UO
        fillna(bool): if True, fill nan values with 50.
    Returns:
        pandas.Series: New feature generated.
    """
    min_l_or_pc = min(tslib.ts_lag(close, 1), low)
    max_h_or_pc = max(tslib.ts_lag(close, 1), high)

    bp = close - min_l_or_pc
    tr = max_h_or_pc - min_l_or_pc

    avg_s = tslib.ts_sum(bp, s) / tslib.ts_sum(tr, s)
    avg_m = tslib.ts_sum(bp, m) / tslib.ts_sum(tr, m)
    avg_l = tslib.ts_sum(bp, l) / tslib.ts_sum(tr, l)

    uo = 100.0 * ((ws * avg_s) + (wm * avg_m) + (wl * avg_l)) / (ws + wm + wl)
    uo = uo.replace([np.inf, -np.inf], np.nan)
    return uo


def stoch(high, low, close, n=14):

    smin = tslib.ts_min(low, n)
    smax = tslib.ts_max(high, n)
    stoch_k = 100 * (close - smin) / (smax - smin)

    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan)
    return stoch_k


def stoch_signal(high, low, close, n=14, d_n=3, fillna=False):
    """Stochastic Oscillator Signal
    Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        d_n(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    stoch_k = stoch(high, low, close, n)
    stoch_d = tslib.ts_mean(stoch_k, d_n)

    stoch_d = stoch_d.replace([np.inf, -np.inf], np.nan)
    return stoch_d


def williams_r(high, low, close, lbp=14, fillna=False):
    """Williams %R
    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
    Developed by Larry Williams, Williams %R is a momentum indicator that is the inverse of the
    Fast Stochastic Oscillator. Also referred to as %R, Williams %R reflects the level of the close
    relative to the highest high for the look-back period. In contrast, the Stochastic Oscillator
    reflects the level of the close relative to the lowest low. %R corrects for the inversion by
    multiplying the raw value by -100. As a result, the Fast Stochastic Oscillator and Williams %R
    produce the exact same lines, only the scaling is different. Williams %R oscillates from 0 to -100.
    Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
    Unsurprisingly, signals derived from the Stochastic Oscillator are also applicable to Williams %R.
    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.
    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces readings from 0 to -20, this indicates
    overbought market conditions. When readings are -80 to -100, it indicates oversold market conditions.
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period
        fillna(bool): if True, fill nan values with -50.
    Returns:
        pandas.Series: New feature generated.
    """

    hh = tslib.ts_max(high, lbp)
    ll = tslib.ts_min(low, lbp)

    wr = -100 * (hh - close) / (hh - ll)

    wr = wr.replace([np.inf, -np.inf], np.nan)

    if fillna:
        wr = wr.fillna(-50)
    return wr


def awesome_oscillator(high, low, s=5, l=34):
    """Awesome Oscillator
    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
    The Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a
    34 Period and 5 Period Simple Moving Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used to affirm trends or to anticipate
    possible reversals.
    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator
    Awesome Oscillator is a 34-period simple moving average, plotted through the central points of the bars (H+L)/2,
    and subtracted from the 5-period simple moving average, graphed across the central points of the bars (H+L)/2.
    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)
    where
    SMA — Simple Moving Average.
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        s(int): short period
        l(int): long period
    Returns:
        pandas.Series: New feature generated.
    """

    mp = 0.5 * (high + low)
    ao = tslib.ts_mean(mp, s) - tslib.ts_mean(mp, l)

    ao = ao.replace([np.inf, -np.inf], np.nan)

    return ao
