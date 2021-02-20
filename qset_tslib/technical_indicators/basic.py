import qset_tslib as tslib


# Bollinger Bands
def bollinger_bands(close, periods, coef=1, min_periods=None):
    """returns [bb_low, bb_mid, bb_high]"""
    c_mean = tslib.ts_mean(close, periods, min_periods=min_periods)
    c_std = tslib.ts_std(close, periods, min_periods=min_periods)
    return c_mean - coef * c_std, c_mean, c_mean + coef * c_std


# Relative Strength Index
def rsi(df, periods=1, min_periods=None, norm=False):
    delta = tslib.ts_diff(df).fillna(0)
    d_up, d_down = delta.copy_path(), delta.copy_path()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0

    avg_up = tslib.ts_mean(d_up, periods, min_periods=min_periods, norm=norm)
    avg_down = tslib.ts_mean(d_down, periods, min_periods=min_periods, norm=norm).abs()

    rs = avg_up / avg_down

    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# Slow Stochastic
def ss(close, high, low, periods=1):
    lowest_low = tslib.ts_min(low, periods=periods)
    highest_high = tslib.ts_max(high, periods=periods)
    return 100.0 * (close - lowest_low) / (highest_high - lowest_low)


# Average directional movement index, Positive/Negative directional indicators
def adx(close, high, low, periods=1):
    up_move = tslib.ts_diff(high)
    down_move = -tslib.ts_diff(low)
    plus_dm = tslib.ifelse((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = tslib.ifelse((up_move < down_move) & (down_move > 0), down_move, 0)

    # True Range
    tr = tslib.max(high - low, abs(high - tslib.ts_lag(close)))
    tr = tslib.max(tr, abs(low - tslib.ts_lag(close)))
    atr = tslib.ts_mean(tr, periods)

    plus_di = 100 * tslib.ts_mean(plus_dm, periods) / atr
    minus_di = 100 * tslib.ts_mean(minus_dm, periods) / atr

    adx = 100 * tslib.ts_mean(abs(plus_di - minus_di), periods) / (plus_di + minus_di)

    return adx, plus_di, minus_di


# On-Balance Volume
def obv(v):
    sign_dv = tslib.sign(tslib.ts_diff(v))
    return (v * sign_dv).cumsum()


def adi(close, low, high, vol, n=288, min_periods=1):
    clv = ((close - low) + (high - close)) / (high - low)
    return tslib.ts_sum(clv * vol, n, min_periods)


def adi_signal(
    close, low, high, vol, n=288, short_sm=288 * 3, long_sm=288 * 10, min_periods=1
):
    acc = adi(close, low, high, vol, n, min_periods)
    signal = tslib.ts_exp_decay(acc, short_sm) - tslib.ts_exp_decay(acc, long_sm)
    return signal


def chaikin_mf(close, low, high, vol, n=288, min_periods=1):
    acc = adi(close, low, high, vol, n, min_periods)
    return acc / tslib.ts_sum(vol, n, min_periods)


def chaikin_mf_signal(
    close, low, high, vol, n=288, short_sm=288 * 3, long_sm=288 * 10, min_periods=1
):
    chk = chaikin_mf(close, low, high, vol, n, min_periods)
    signal = tslib.ts_exp_decay(chk, short_sm) - tslib.ts_exp_decay(chk, long_sm)
    return signal


def force_index(close, vol, n=288, min_periods=1):
    fi = tslib.ts_sum(tslib.ts_diff(close, 1) * vol, n, min_periods)
    return fi / tslib.ts_sum(vol, n, min_periods)


def ease_of_movement(high, low, vol, n=288, dist_lag=1, min_periods=1):
    dist_moved = (high + low) / 2 - tslib.ts_lag(high + low, dist_lag) / 2
    box_ratio = vol / (high - low)
    emv = dist_moved / box_ratio
    return tslib.ts_mean(emv, n, min_periods=min_periods)


def vpt(close, vol, n=288, min_periods=1):
    vpt = tslib.ts_sum(vol * tslib.ts_returns(close, 1), n, min_periods)
    return vpt


def vpt_on_imbalance(close, bv, sv, n=288, min_periods=1):
    vpt_on_imb = tslib.ts_sum(
        tslib.ifelse(
            (bv - sv) * tslib.ts_returns(close, 1) > 0,
            (bv - sv) * tslib.ts_returns(close, 1),
            tslib.make_like(close, 0),
        ),
        n,
        min_periods,
    )
    return vpt_on_imb
