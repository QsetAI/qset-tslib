from qset_tslib.dataseries import *


# Bollinger Bands
def bollinger_bands(close, periods, coef=1, min_periods=None):
    """returns [bb_low, bb_mid, bb_high]"""
    c_mean = ts_mean(close, periods, min_periods=min_periods)
    c_std = ts_std(close, periods, min_periods=min_periods)
    return c_mean - coef * c_std, c_mean, c_mean + coef * c_std


# Relative Strength Index
def rsi(df, periods=1, min_periods=None, norm=False):
    delta = ts_diff(df).fillna(0)
    d_up, d_down = delta.copy_path(), delta.copy_path()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0

    avg_up = ts_mean(d_up, periods, min_periods=min_periods, norm=norm)
    avg_down = ts_mean(d_down, periods, min_periods=min_periods, norm=norm).abs()

    rs = avg_up / avg_down

    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# Slow Stochastic
def ss(close, high, low, periods=1):
    lowest_low = ts_min(low, periods=periods)
    highest_high = ts_max(high, periods=periods)
    return 100.0 * (close - lowest_low) / (highest_high - lowest_low)


# Average directional movement index, Positive/Negative directional indicators
def adx(close, high, low, periods=1):
    up_move = ts_diff(high)
    down_move = - ts_diff(low)
    plus_dm = ifelse(op_and(up_move > down_move, up_move > 0), up_move, 0)
    minus_dm = ifelse(op_and(up_move < down_move, down_move > 0), down_move, 0)

    # True Range
    tr = op_max(high - low, abs(high - ts_lag(close)))
    tr = op_max(tr, abs(low - ts_lag(close)))
    atr = ts_mean(tr, periods)

    plus_di = 100 * ts_mean(plus_dm, periods) / atr
    minus_di = 100 * ts_mean(minus_dm, periods) / atr

    adx = 100 * ts_mean(abs(plus_di - minus_di), periods) / (plus_di + minus_di)

    return adx, plus_di, minus_di


# On-Balance Volume
def obv(v):
    sign_dv = sign(ts_diff(v))
    return (v * sign_dv).cumsum()


def adi(close, low, high, vol, n=288, min_periods=1):
    clv = ((close-low)+(high-close))/(high-low)
    return ts_sum(clv*vol, n, min_periods)


def adi_signal(close, low, high,vol, n=288, short_sm=288*3, long_sm = 288*10, min_periods=1):
    acc = adi(close, low, high, vol, n, min_periods)
    signal = ts_exp_decay(acc, short_sm) - ts_exp_decay(acc, long_sm)
    return signal


def chaikin_mf(close, low, high, vol, n=288, min_periods=1):
    acc = adi(close, low, high, vol, n, min_periods)
    return acc / ts_sum(vol, n, min_periods)


def chaikin_mf_signal(close, low, high, vol, n=288, short_sm=288*3, long_sm = 288*10, min_periods=1):
    chk = chaikin_mf(close, low, high, vol, n, min_periods)
    signal = ts_exp_decay(chk, short_sm) - ts_exp_decay(chk, long_sm)
    return signal


def force_index(close, vol, n=288, min_periods=1):
    fi = ts_sum(ts_diff(close, 1) * vol, n, min_periods)
    return fi / ts_sum(vol, n, min_periods)


def ease_of_movement(high, low, vol, n=288, dist_lag=1, min_periods=1):
    dist_moved = ((high+low)/2 - ts_lag(high+low, dist_lag)/2)
    box_ratio = vol/(high-low)
    emv = dist_moved/box_ratio
    return ts_mean(emv, n, min_periods=min_periods)


def vpt(close, vol, n=288, min_periods=1):
    vpt = ts_sum(vol * ts_returns(close, 1), n, min_periods)
    return vpt


def vpt_on_imbalance(close, bv, sv, n=288, min_periods=1):
    vpt_on_imb = ts_sum(ifelse((bv-sv)*ts_returns(close,1) > 0, (bv-sv)*ts_returns(close,1), constant(0, close)),
                        n, min_periods)
    return vpt_on_imb
