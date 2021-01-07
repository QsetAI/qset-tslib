import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from qset_tslib.dataseries import constant
from sklearn.linear_model import Ridge, LinearRegression


class RollingBetaCalc:
    def __init__(self, orig_X, orig_y, calc_score=False):
        if not (orig_X.index == orig_y.index).all():
            raise ValueError
        self.y = orig_y
        self.X = orig_X
        self.calc_score = calc_score

        mask = (~self.y.isnull()) & ~(self.X.isnull().any(axis=1))
        self.clean_X = self.X.loc[mask]
        self.clean_y = self.y.loc[mask]
        self.result = []
        self.score = []
        self.res_indices = []

    def rolling_reg(self, window, **kwargs):
        if len(self.clean_X) < window:
            return self.result
        tmp_ind = pd.Series(data=range(len(self.clean_y) - 1))
        tmp_res = tmp_ind.rolling(window).apply(
            lambda ii: self.reg(self.clean_X.iloc[ii], self.clean_y.iloc[ii], int(ii[len(ii) - 1] + 1), **kwargs))
        self.finalize()
        return self.result

    def reg(self, X, y, num_index, **kwargs):
        pass

    #@profile
    def finalize(self):
        if self.calc_score:
            self.score = pd.DataFrame(data=self.score, index = self.res_indices, columns=['score'])
            self.score = self.score.reindex(self.y.index, axis=0)
        self.result = pd.DataFrame(data=self.result, index = self.res_indices, columns=self.X.columns)
        self.result = self.result.reindex(self.X.index, axis=0)


#alpha does what is supposed to do in ridge
class RidgeBetaCalc(RollingBetaCalc):

    #@profile
    def reg(self, X, y, num_index, **kwargs):
        alpha_coef = kwargs.get('alpha', 0.05)
        normalize = kwargs.get('normalize', True)
        fit_intercept = kwargs.get('fit_intercept', False)
        if normalize:
            new_x = (X-X.mean(axis=0))/X.std(axis=0)
            new_y = (y-y.mean())/y.std()
        else:
            new_x = X
            new_y = y
        model = Ridge(alpha=alpha_coef, fit_intercept=fit_intercept, normalize=normalize)
        model.fit(new_x, new_y)
        orig_index = self.y.index[num_index]
        self.result.append(model.coef_)
        self.res_indices.append(orig_index)
        #self.result.at[orig_index] = model.coef_
        if self.calc_score:
            self.score.append(model.score(X,y))
            #self.score.at[orig_index] = model.score(X, y)
        return -1

##alpha is not used
class OLSBetaCalc(RollingBetaCalc):

    #@profile
    def reg(self, X, y, num_index, **kwargs):
        fit_intercept = kwargs.get('fit_intercept', False)
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)
        orig_index = self.y.index[num_index]
        #self.result.at[orig_index] = model.coef
        self.result.append(model.coef_)
        self.res_indices.append(orig_index)

        if self.calc_score:
            self.score.append(model.score(X, y))
            #self.score.at[orig_index] = model.score(X, y)
        return -1


class QuantileBetaCalc(RollingBetaCalc):
    def reg(self, X, y, num_index, **kwargs):
        model = QuantReg(X, y)
        q = kwargs['quantile']
        model.fit(q)



