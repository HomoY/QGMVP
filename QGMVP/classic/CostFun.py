"""A class for cost functions"""

__author__ = "HMY"
__date__ = "2023-01-24"

import numpy as np

# from Util import spendtime
import copy


def norma(x: np.ndarray) -> float:
    """renormalized x when input"""
    return x / np.sum(x)


class costsfun(object):
    """
    A Class for classical function of risk aversion and risk parity.

    """

    def __init__(
        self,
        sigma: np.ndarray,
        mu: np.ndarray = None,
        lamb: float = None,
        sum_val: float = None,
    ):
        """
        classical function

        x: ratio/weight of asset investment, the range is depends on the input, sometimes not normalized as 0-1, incase not renormalized, we normalized it as initial
        lamb: The parameter describing the risk aversion of the investor.
        sigma: The estimated covariance matrix between assets i and j.
        mu: The estimated return vector for different assets.
        name: the name of function, srp: succsive risk parity model
        """
        self.n = len(sigma)  # asset sets
        self.lamb = lamb
        self.sigma = sigma
        self.mu = mu
        self.sum_val = sum_val

    def _costtest(self, x: np.ndarray) -> float:
        """
        A simple function used to test classical optimizer

        Args:
            x (np.ndarray): inputs
        Returns:
            _type_: _description_
        """

        return x @ x.T

    def var(self, x: np.ndarray) -> float:
        """
        A simple function to get the variance

        Args:
            x (np.ndarray): inputs
        Returns:
            _type_: _description_
        """

        return x @ self.sigma @ x.T

    # @ nb.jit(nopython=True)
    def _costFun(self, x: np.ndarray) -> float:
        """
        [risk return classical function]
            fun =  0.5 x@sigma@ x -lamb * mu@x

        Args:
            x: weight of asset, the range is from 0 to 1
            lamb: The parameter describing the risk aversion of the investor.
            sigma: The estimated covariance matrix between assets i and j.
            mu: The estimated return vector for different assets.

        Return:
            risk return function value with input x
        """

        var = 0.5 * x @ self.sigma @ x.T
        exc = self.lamb * (x.T @ self.mu)

        return var - exc

