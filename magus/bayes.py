from __future__ import print_function, division
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi', 'lcb']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'lcb':
            return self._lcb(x, gp, self.kappa)


    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    @staticmethod
    def _lcb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean - kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)

def GP_fit(x0, y0, gp_params=None, alpha=1e-5):
    """
    Return a gaussian process fitted to the input data
    """

    xp = np.array(x0)
    yp = np.array(y0)

    if gp_params is not None:
        model = GaussianProcessRegressor(**gp_params)
    else:
        kernel = kernels.DotProduct(sigma_0=0)
        kernel = kernels.Exponentiation(kernel, 2)
        model = GaussianProcessRegressor(kernel=kernel,
                                         alpha=alpha,
                                         n_restarts_optimizer=25,
                                         normalize_y=False)
    model.fit(xp, yp)

    return model

def atoms_util(fp, util, gp, symbols, y_max):
    """
    Calculate the results of Utility Function.
    """
    return util.utility(x=fp.reshape(1, -1), gp=gp, y_max=y_max)
