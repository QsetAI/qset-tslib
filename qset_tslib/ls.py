import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import Ridge, LinearRegression


def ts_ls_slope(df_x, df_y, window, min_periods=2, add_notna_mask=False, ddof=1):
    """Calculates running slope of simple linear regression between windows of dataframes df_x and df_y, where df_y depends on df_x."""

    if add_notna_mask:
        _mask = df_x.notnull() & df_y.notnull()
        df_x = df_x.where(_mask)
        df_y = df_y.where(_mask)

    cov_xy = (df_x * df_y).rolling(
        window, min_periods=min_periods
    ).mean() - df_x.rolling(window, min_periods=min_periods).mean() * df_y.rolling(
        window, min_periods=min_periods
    ).mean()
    var_x = df_x.rolling(window, min_periods=min_periods).var(ddof=ddof)
    return cov_xy / var_x


def ts_ls_slope_by_timeline(df, *args, **kwargs):
    """Calculates rolling_ls_slope for every columns in dataframe as df_y and with range(len(df)) as df_x."""
    n_cols = len(df.columns)
    df_x = pd.DataFrame(
        np.repeat([np.arange(len(df))], n_cols, axis=0).T,
        index=df.index,
        columns=df.columns,
    )
    df_y = df
    return ts_ls_slope(df_x, df_y, *args, **kwargs)


class RollingBetaCalc:
    def __init__(self, x, y, calc_score=False):
        assert (x.index == y.index).all()

        self.x = x
        self.y = y
        self.calc_score = calc_score

        mask = (~self.y.isnull()) & ~(self.x.isnull().any(axis=1))
        self.clean_x = self.x.loc[mask]
        self.clean_y = self.y.loc[mask]

        self.result = []
        self.score = []
        self.res_indices = []

    def rolling_reg(self, window, **kwargs):
        if len(self.clean_x) < window:
            return self.result
        tmp_ind = pd.Series(data=range(len(self.clean_y) - 1))
        tmp_res = tmp_ind.rolling(window).apply(
            lambda ii: self.reg(
                self.clean_x.iloc[ii],
                self.clean_y.iloc[ii],
                int(ii[len(ii) - 1] + 1),
                **kwargs
            )
        )
        self.finalize()
        return self.result

    def reg(self, X, y, num_index, **kwargs):
        pass

    def finalize(self):
        if self.calc_score:
            self.score = pd.DataFrame(
                data=self.score, index=self.res_indices, columns=["score"]
            )
            self.score = self.score.reindex(self.y.index, axis=0)
        self.result = pd.DataFrame(
            data=self.result, index=self.res_indices, columns=self.x.columns
        )
        self.result = self.result.reindex(self.x.index, axis=0)


# alpha does what is supposed to do in ridge
class RidgeBetaCalc(RollingBetaCalc):
    def reg(self, X, y, num_index, **kwargs):
        alpha_coef = kwargs.get("alpha", 0.05)
        normalize = kwargs.get("normalize", True)
        fit_intercept = kwargs.get("fit_intercept", False)
        if normalize:
            new_x = (X - X.mean(axis=0)) / X.std(axis=0)
            new_y = (y - y.mean()) / y.std()
        else:
            new_x = X
            new_y = y
        model = Ridge(
            alpha=alpha_coef, fit_intercept=fit_intercept, normalize=normalize
        )
        model.fit(new_x, new_y)
        orig_index = self.y.index[num_index]
        self.result.append(model.coef_)
        self.res_indices.append(orig_index)
        # self.result.at[orig_index] = model.coef_
        if self.calc_score:
            self.score.append(model.score(X, y))
            # self.score.at[orig_index] = model.score(X, y)
        return -1


##alpha is not used
class OLSBetaCalc(RollingBetaCalc):
    def reg(self, X, y, num_index, **kwargs):
        fit_intercept = kwargs.get("fit_intercept", False)
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)
        orig_index = self.y.index[num_index]
        # self.result.at[orig_index] = model.coef
        self.result.append(model.coef_)
        self.res_indices.append(orig_index)

        if self.calc_score:
            self.score.append(model.score(X, y))
            # self.score.at[orig_index] = model.score(X, y)
        return -1


class QuantileBetaCalc(RollingBetaCalc):
    def reg(self, X, y, num_index, **kwargs):
        model = QuantReg(X, y)
        q = kwargs["quantile"]
        model.fit(q)


# todo: test rolling regression classes
