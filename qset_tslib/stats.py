import numpy as np
import pandas as pd
import scipy.stats
import math

import qset_tslib as tslib
from datetime import datetime, timedelta


def _get_timeframe(td):
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds // 60) % 60
    seconds = td.seconds - (3600 * hours + 60 * minutes)

    res = ""
    if days > 0:
        res += f"{days}d"
    if hours > 0:
        res += f"{hours}h"
    if minutes > 0:
        res += f"{minutes}m"

    return res


def calc_basic_stats(returns, periods=365):
    vola_yearly = np.sqrt(periods) * np.std(returns)

    return_yearly = np.mean(returns) * periods

    sharpe = return_yearly / vola_yearly if vola_yearly > 0 else 0

    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_vola = np.std(downside_returns)
    downside_vola_yearly = downside_vola * np.sqrt(periods)

    sortino = return_yearly / downside_vola_yearly if downside_vola_yearly > 0 else 0

    net_profit = np.sum(returns)

    skewness = scipy.stats.skew(returns)
    try:
        skewness = skewness[0]
    except:
        # case of [skewness]
        pass

    kurtosis = scipy.stats.kurtosis(returns)
    try:
        kurtosis = kurtosis[0]
    except:
        # case of [kurtosis]
        pass
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "return_yearly": return_yearly,
        "vola_yearly": vola_yearly,
        "net_profit": net_profit,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def calc_t_stat(returns):
    returns = np.array(returns).astype(float)

    if np.sum(np.isfinite(returns)) == 0:
        return 0

    first_non_zero = next((i for i, x in enumerate(returns) if x), None)
    returns[0:first_non_zero] = np.nan
    return (
        np.nanmean(returns) / np.nanstd(returns) * np.sqrt(np.sum(np.isfinite(returns)))
    )


def calc_dd_stats(pnl):
    """
    :param pnl: `np.array`
    :return: Drawdown stats
    """
    HH = np.maximum.accumulate(pnl)
    if np.array_equal(HH, pnl):
        max_dd_beg = 0
        max_dd_end = 0
        max_dd = 0.0
        max_off_peak = 0
        max_off_peak_beg = 0
        max_off_peak_end = 0
    else:
        underwater = pnl / HH
        max_dd_end = tslib.argmin(underwater, last=True)

        max_dd = 1 - np.min(underwater)
        max_dd_beg = np.argmax(pnl[0:max_dd_end])

        is_underwater = pnl < HH
        is_underwater = np.insert(is_underwater, [0, len(is_underwater)], False)
        underwater_diff = np.diff(is_underwater.astype("int"))
        time_off_peak_beg = np.where(underwater_diff == 1)[0]
        time_off_peak_end = np.where(underwater_diff == -1)[0]

        time_off_peak = time_off_peak_end - time_off_peak_beg

        if len(time_off_peak) != 0:
            max_off_peak = np.max(time_off_peak)
            max_time_off_peak_idx = np.argmax(time_off_peak)
            max_off_peak_beg = time_off_peak_beg[max_time_off_peak_idx] - 1
            max_off_peak_end = time_off_peak_end[max_time_off_peak_idx]
        else:
            max_off_peak = 0
            max_off_peak_beg = 0
            max_off_peak_end = 0

    return {
        "max_dd": max_dd,
        "max_dd_beg": max_dd_beg,
        "max_dd_end": max_dd_end,
        "max_off_peak": max_off_peak,
        "max_off_peak_beg": max_off_peak_beg,
        "max_off_peak_end": max_off_peak_end,
    }


def calc_stats(returns, turnover=None, booksize=1.0, days_in_a_year=365):
    index = returns.index
    turnover = turnover if turnover is not None else pd.Series(0.0, index=index)

    stats = {}
    stats["begin"] = index[0]
    stats["end"] = index[-1]

    for key in ["begin", "end"]:
        try:
            stats[key] = stats[key].to_pydatetime()
        except:
            pass

    stats["periods"] = len(returns)
    stats["days_in_a_year"] = days_in_a_year

    # convert to numpy array
    returns = returns.values
    turnover = turnover.values

    if len(returns) <= 1:
        return stats

    stats.update(calc_basic_stats(returns, periods=days_in_a_year))

    pnl = (1 + returns).cumprod()
    stats.update(calc_dd_stats(pnl))

    stats["mar"] = stats["return_yearly"] / stats["max_dd"]
    stats["t_stat"] = calc_t_stat(returns)
    stats["margin"] = (
        returns.sum() / turnover.sum() * 10000 if turnover.sum() != 0 else np.inf
    )
    stats["turnover"] = turnover.mean()
    stats["turnover_adj"] = (turnover / booksize).mean()
    stats["timeframe"] = _get_timeframe(index[1] - index[0])
    return stats


def test():
    def datetime_range(beg, period=None, end=None, n=None):
        if end:
            while beg < end:
                yield beg
                beg = min(beg + period, end)
        elif n:
            for i in range(n):
                yield beg + i * period
        else:
            raise Exception("end or n not specified")

    df = pd.DataFrame(
        np.random.uniform(-0.01, 0.012, 100),
        index=datetime_range(datetime(2020, 1, 20), period=timedelta(days=1), n=100),
    )
    print(df)
    from pprint import pprint

    pprint(calc_stats(df))


if __name__ == "__main__":
    test()
