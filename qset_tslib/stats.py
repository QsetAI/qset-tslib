import numpy as np
import pandas as pd
import scipy.stats
import math

import qset_tslib as tslib
from datetime import datetime, timedelta
from utils_ak.time import cast_sec


# todo: refactor
# todo: test


def calc_batch_stats(returns, costs=None):
    """
    :param returns: `pd.DataFrame`. Returns for different alphas in one dataframe
    :return: `pd.DataFrame`. Stats dataframe
    """
    stats_calc = StatsCalculator()
    costs_key = "with_costs" if costs is not None else "without_costs"

    res = []
    keys = []
    for alpha in returns.columns:

        if costs is None:
            stats = stats_calc.collect(returns[alpha])[costs_key]
        else:
            stats = stats_calc.collect(returns[alpha], costs[alpha])[costs_key]
        if not keys:
            keys = list(stats.keys())
        res.append([alpha] + [stats[key] for key in keys])
    df = pd.DataFrame(res, columns=["alpha"] + keys)
    df.pop("beg")
    df.pop("end")
    df = df.set_index("alpha")

    df["timeframe"] = df["timeframe"].apply(cast_sec)
    # todo: make properly
    df["vola_yearly"] = -df["vola_yearly"]
    df["max_dd"] = -df["max_dd"]
    df["max_off_peak"] = -df["max_off_peak"]
    return df


class StatsCalculator(object):
    @staticmethod
    def collect(pnl, costs=None, turnover=None, booksize=1.0, year_days=365):
        """ Collect stats with and without costs, daily and non-daily"""
        index = pnl.index
        costs = costs if costs is not None else pd.Series(0, index=index)
        turnover = turnover if turnover is not None else pd.Series(0.0, index=index)

        # also calc daily stats
        daily_df = pd.DataFrame(
            {"pnl": pnl, "costs": costs, "turnover": turnover, "booksize": booksize}
        )
        daily_df["date"] = pd.Series(daily_df.index).dt.date.values
        daily_df = daily_df.groupby("date").agg(
            {"pnl": np.sum, "costs": np.sum, "turnover": np.sum, "booksize": np.mean}
        )

        # convert index from date to datetime
        daily_df.index = pd.to_datetime(daily_df.index)

        periods_per_day = int(math.ceil(len(index) / len(daily_df)))

        stats_costs = StatsCalculator.calc(
            pnl - costs, costs, turnover, booksize, year_days * periods_per_day
        )
        stats_no_costs = StatsCalculator.calc(
            pnl, costs, turnover, booksize, year_days * periods_per_day
        )
        daily_stats_costs = StatsCalculator.calc(
            daily_df["pnl"] - daily_df["costs"],
            daily_df["costs"],
            daily_df["turnover"],
            daily_df["booksize"],
            year_days,
        )
        daily_stats_no_costs = StatsCalculator.calc(
            daily_df["pnl"],
            daily_df["costs"],
            daily_df["turnover"],
            daily_df["booksize"],
            year_days,
        )

        res = {"with_costs": daily_stats_costs, "without_costs": daily_stats_no_costs}

        # todo: make properly
        # replace daily stats with period stats if one day is backtested
        if "sharpe" not in daily_stats_costs:
            res["with_costs"] = stats_costs
            res["without_costs"] = stats_no_costs

        res["with_costs"]["sharpe_step"] = stats_costs["sharpe"]
        res["with_costs"]["turnover_step"] = stats_costs["turnover"]
        res["with_costs"]["turnover_adj_step"] = stats_costs["turnover_adj"]
        res["with_costs"]["timeframe"] = stats_costs["timeframe"]

        res["without_costs"]["sharpe_step"] = stats_no_costs["sharpe"]
        res["without_costs"]["turnover_step"] = stats_no_costs["turnover"]
        res["without_costs"]["turnover_adj_step"] = stats_no_costs["turnover_adj"]
        res["without_costs"]["timeframe"] = stats_no_costs["timeframe"]

        return res

    @staticmethod
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

    @staticmethod
    def calc(pnl, costs=None, turnover=None, booksize=1.0, n_periods=365):
        """
        Calc stats for specified series
        :param n_periods: int
        :return: dict
        """
        index = pnl.index
        costs = costs if costs is not None else pd.Series(0, index=index)
        turnover = turnover if turnover is not None else pd.Series(0.0, index=index)

        stats = dict()
        stats["beg"] = index[0]
        stats["end"] = index[-1]
        stats["periods"] = len(pnl)
        stats["n_periods"] = n_periods

        returns = pnl / booksize

        # convert to numpy array
        returns = returns.values
        costs = costs.values
        turnover = turnover.values

        if len(returns) <= 1:
            return stats

        basic_stats = StatsCalculator.calc_basic_stats(returns, n_periods=n_periods)
        stats.update(basic_stats)

        portfolio_size = (1 + returns).cumprod()
        dd_stats = StatsCalculator.calc_dd_stats(portfolio_size)
        stats.update(dd_stats)

        stats["mar"] = basic_stats["return_yearly"] / dd_stats["max_dd"]
        stats["t_stat"] = StatsCalculator.t_stat(returns)
        stats["costs_ratio"] = (
            returns.sum() / costs.sum() if costs.sum() != 0 else np.inf
        )
        stats["margin"] = (
            returns.sum() / turnover.sum() * 10000 if turnover.sum() != 0 else np.inf
        )
        stats["turnover"] = turnover.mean()
        stats["turnover_adj"] = (turnover / booksize).mean()
        stats["timeframe"] = StatsCalculator._get_timeframe(index[1] - index[0])
        return stats

    @staticmethod
    def calc_basic_stats(returns, n_periods=365):
        vola_yearly = np.sqrt(n_periods) * np.std(returns)

        return_yearly = np.mean(returns) * n_periods

        sharpe = return_yearly / vola_yearly if vola_yearly > 0 else 0

        downside_returns = returns.copy_path()
        downside_returns[downside_returns > 0] = 0
        downside_vola = np.std(downside_returns)
        downside_vola_yearly = downside_vola * np.sqrt(n_periods)

        sortino = (
            return_yearly / downside_vola_yearly if downside_vola_yearly > 0 else 0
        )

        net_profit = np.sum(returns)

        skewness = scipy.stats.skew(returns)
        kurtosis = scipy.stats.kurtosis(returns)
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "return_yearly": return_yearly,
            "vola_yearly": vola_yearly,
            "net_profit": net_profit,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    @staticmethod
    def calc_dd_stats(portfolio_size):
        """
        :param portfolio_size: `np.array`
        :return: Drawdown stats
        """
        HH = np.maximum.accumulate(portfolio_size)
        if np.array_equal(HH, portfolio_size):
            max_dd_beg = 0
            max_dd_end = 0
            max_dd = 0.0
            max_off_peak = 0
            max_off_peak_beg = 0
            max_off_peak_end = 0
        else:
            underwater = portfolio_size / HH
            max_dd_end = tslib.argmin(underwater, last=True)

            max_dd = 1 - np.min(underwater)
            max_dd_beg = np.argmax(portfolio_size[0:max_dd_end])

            is_underwater = portfolio_size < HH
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

    @staticmethod
    def t_stat(returns):
        returns = np.array(returns).astype(float)

        if np.sum(np.isfinite(returns)) == 0:
            return 0

        first_non_zero = next((i for i, x in enumerate(returns) if x), None)
        returns[0:first_non_zero] = np.nan
        return (
            np.nanmean(returns)
            / np.nanstd(returns)
            * np.sqrt(np.sum(np.isfinite(returns)))
        )
