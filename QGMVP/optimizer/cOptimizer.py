# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""A Class for classical optimizors"""
__author__ = "HMY"
__date__ = "2023-02-08"


from typing import Callable, Tuple, Union, List

import numpy as np
from QGMVP.utils.Util import *
import numba as nb
import scipy.optimize as optz


class cOptimizer(object):
    """
    The class for classical optimizor for our QAOA problem.

    """

    def __init__(self):
        """
        Class initialization.

        """
        pass

    def annealing(
        self,
        objFun: Callable[..., float],
        initParam: list,
        bounds: optz.Bounds,
        numIter: int = False,
        **_,
    ) -> Tuple[np.array, float]:
        """
        Using the annealing method to search for the global minimum. This method does not require information on derivatives.

        initParam: A list that is [<initGamma>] + [<initBeta>].
        numIter: The number of iteration of objective function evaluation.
        restart_temp_ratio

        Return: The best possible parameters found by the annealing method, as well as the corresponding optimal function value.

        """
        print("optimize with annealing...")
        callback_results = []

        def callback(x, f, context):
            callback_results.append((context, x.copy(), f))
            print(f"Iteration: {context}, x: {x}, f: {f}")

        if numIter != False:
            optRes = optz.dual_annealing(
                func=objFun,
                x0=np.array(initParam),
                bounds=bounds,
                maxfun=numIter,
            )
        else:
            optRes = optz.dual_annealing(
                func=objFun,
                x0=np.array(initParam),
                bounds=bounds,
            )
        print(optRes)
        params = optRes.get("x")
        val = optRes.get("fun")

        return np.array(params), val

    def COBYLA(
        self,
        objFun: Callable[..., float],
        initParam: list,
        lower: List[float],
        upper: List[float],
        numIter: int,
        verbose: bool,
        tol: float = 1e-6,
        **_,
    ):
        """
        Using the COBYLA method to search for the local minimum. This method does not require information on derivatives.

        objFun: The objective function to be optimized.
        initParam: A list that is [<initGamma>] + [<initBeta>].
        numIter: The number of iteration of objective function evaluation.
        tol: The tolerance for the gradient projected on certain point to make a stop.
        verbose: Whether print useful information or not.

        Return: The best possible parameters found by using COBYLA method, as well as the corresponding optimal function value.

        """
        print("optimize with Cobyla...")
        optRes = optz.minimize(
            fun=objFun,
            x0=np.array(initParam),
            method="COBYLA",
            constraints=optz.LinearConstraint(
                A=np.diag(np.ones(len(initParam))), lb=lower, ub=upper
            ),
            options={
                "rhobeg": 1.0,
                "maxiter": numIter,
                "disp": False,
                "tol": tol,
                "catol": 0,
            },
        )
        params = optRes.get("x")
        val = optRes.get("fun")
        if verbose == True:
            print(optRes)

        return np.array(params), val


@nb.jit(nopython=True, nogil=True)
def bruteForceHelper(m: int, n: int, modArray: np.ndarray):
    """
    Given a number representing the asset position, decode the number, return the corresponding position and the binary string.

    m: The number representing the position.
    n: The total number of assets.
    modArray: The array of powers of 3.

    Return: The asset position array and the binary number array, as well as the net investment.

    """
    binNum = 0
    assetPos = np.zeros(n, dtype=np.float64)
    d = 0
    for i in range(n):
        pos = m // modArray[i] % 3 - 1
        assetPos[i] = pos
        d += pos
        if pos == 1:
            binNum = (binNum << 2) | 1
        elif pos == -1:
            binNum = (binNum << 2) | 2
        else:
            binNum = binNum << 2

    return assetPos, binNum, d
