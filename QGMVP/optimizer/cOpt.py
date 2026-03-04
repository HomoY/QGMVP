# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""classical optimization methods."""
__author__ = "HMY"
__date__ = "2023-02-01"

from typing import Callable, Tuple, Union

import numpy as np
from tqdm import tqdm
from QGMVP import *
from QGMVP.classic.CostFun import *
from cvxopt import matrix, solvers
from QGMVP.optimizer.cOptimizer import cOptimizer


"""
    Get the parameter for the quadratic optimization, the standard of optimization form is
    min 0.5 x^T P x +q^T x
    s.t. Gx < h
         Ax = b
"""


def cpo(
    n: int,
    sigma: np.ndarray,
    lamb: float = 1.0,
    mu: np.ndarray = None,
    d: np.ndarray = None,
    f: np.ndarray = None,
    a: float = 1.0,
    **kwargs,
) -> list:
    """
    Find the optimal solution for the Continuous relaxation of our Portfolio Optimization problem. You could pass me the parameters in the most original model, i.e., the proportion represented by each qubit, a.

    the model is:
        Sum_ij lamb * a^2 *sigma_ij (di+xi)(dj+xj) -(1-lamb) *a * Sum (di +xi) *ri
    =  Sum_ij lamb *a^2*sigma_ij * xi *xj +2 lamb *a^2 Sumi (Sumj sigmaij *dj) *xi -(1-lamd)*a*Sumi xi*ri + Constant

    st: Sum(di +xi) *a = 1
        wi in [bi, Bi]
    * very import, you need to guarantee you code can find the smallest, the glbal pahse need to be lagger than 0
    where wi = (di+xi) *a, bi = di*a, and Bi = (di+fi)*a
    that means xi in [0, fi]

    n (int): assets size.
    sigma: The covariance matrix.
    lamb: The risk aversion coefficient. Default is 1
    mu: The expected return vector. Default is None, and will be translated as 0 array
    d: The lower bound for each asset position. Default is None, be translated as 0 array
    f: <The upper bound minus the lower bound> for each asset position. Default is None, be translated as 1 array
    a: The proportion each qubit represents, shall be like 0.1 or 0.05, etc. the default is 1.0, which means no amplification
    * when all default is choosen become optimization only cares about the risk

    return: args (list): contains parameters of optimization [P, q, G, h, A, b]
    """

    if d == None:
        d = np.zeros(n)
    if f == None:
        f = np.ones(n)
    if isinstance(mu, type(None)):
        mu = np.zeros(n)

    newP = 2.0 * lamb * a * a * sigma
    newP = 0.5 * (newP + newP.T)  # make sure P is symmetric
    newMu = -(1.0 - lamb) * a * mu
    newMu += 2.0 * lamb * a * a * sigma @ d

    P = matrix(newP * 1.0)
    q = matrix(newMu * 1.0)
    A = matrix(np.ones(n), (1, n))
    b = matrix(1 / a - np.sum(d))
    I = np.diag(np.ones(shape=n))
    G = matrix(np.vstack((I, -I)))
    h = matrix(np.concatenate((f, np.zeros(shape=n))))

    args = [P, q, G, h, A, b]

    return args


def scrip(
    n: int,
    Qk: np.ndarray = None,
    qk: np.ndarray = None,
    **kwargs,
):
    """
    Get the qudratic optimization parameter

    function is:
    f(x) = 0.5 * x * Qk *x + x *qk
    s.t. x*1 =1, x in [0,1]

    n (int): assets size.
    Qk (np.ndarray): the covariance matrix
    qk (np.ndarray): the single terms

    """
    # This part is used for initialize the optimization
    G = matrix(
        np.append(np.zeros([n, n]) - np.identity(n), np.identity(n), axis=0)
    )  # get a long vector for the restriction
    h = matrix(np.append(np.zeros(n), np.ones(n)))
    # linear constraint, if we conly consider sum is equal to 1
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)
    args = [G, h, A, b]

    if not isinstance(qk, type(None)):
        q = matrix(qk * 1.0)
        args.insert(0, q)
    if not isinstance(Qk, type(None)):
        P = matrix(Qk * 1.0)
        args.insert(0, P)

    return args


def quadOpt(P, q, G=None, h=None, A=None, b=None, **kwargs) -> np.ndarray:
    """
    Perform the quadratic Optimization
    args (list): [P, q, G, h, A, b] parameters to perform the qudratic optimization, the parameter is from

    min 0.5 x^T P x +q^T x
    s.t. Gx < h
    Ax = b
    """
    # if kwargs["verbose"]:
    print("\nStart classical quadratic programming...")
    solvers.options["show_progress"] = kwargs["verbose"]
    sol = solvers.qp(P, q, G, h, A, b)
    optX = np.squeeze(np.array(sol["x"]))
    if kwargs["verbose"]:
        print("The optimal solution is:", optX, "and f(x):", sol["primal objective"])
    print("End classical quadratic programming.")

    return optX


def quadOptInitial(P, q, G=None, h=None, A=None, b=None) -> np.ndarray:
    """
    Perform the quadratic Optimization for initial state optimization
    args (list): [P, q, G, h, A, b] parameters to perform the qudratic optimization, the parameter is from

    min 0.5 x^T P x +q^T x
    s.t. Gx < h
    Ax = b
    """
    print("\nStart classical quadratic programming...")
    solvers.options["show_progress"] = False
    sol = solvers.qp(P, q, G, h, A, b)
    optX = np.squeeze(np.array(sol["x"]))

    return optX


def bruteForce(
    costFun: Callable[..., float],
    n: int,
    bitRan: np.ndarray,
    a: np.ndarray,
    D: int,
    memory: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float, float]:
    """


    Find the global optimal value of the cost function of interest.

    costFun: The cost function of interest.
    n: The number of assets.
    bitRan (np.ndarray): the range of each asset change by 1 increase like for allo is [5,5,5,5,5], then bitRan is [32,32,32,32,32]
    a: the amplitude of each qubit blocks
    D (int): the constraint of investment, default is 1
    memory: Whether save the cost function result or not.
    verbose: Whether print useful information or not.

    Return: The binary string representing the best asset allocation scheme, as well as the correpsonding cost function value. Also return the dictionary containing a dictionary with bit string as the key and cost as the value.

    """
    divisors = np.array([int(np.prod(bitRan[:x])) for x in range(0, n + 1)])
    num = divisors[n]  # The last one is the number of the total calculation need
    quRatio = []  # to record asset ratio
    res = []  # to record the value
    reNum = []  # to record number
    deg_m = []  # to record degeneracy for the minimum
    deg_g = []  # to record degeneracy for the greatest

    with tqdm(
        total=num, leave=False, disable=not verbose
    ) as pBar:  # this is the status for process bar
        pBar.set_description("Brute Force Traversing...")
        bestPos = None
        worstPos = None
        bestCost = np.inf  # first set the best cost as a engh big number
        worstCost = -np.inf
        for i in range(num):
            pBar.update(1)
            assetX = np.zeros(n)
            for j in range(n):
                assetX[j] = i // divisors[j] % bitRan[j]
            sumBits = np.sum(assetX, dtype=np.longlong)
            assetX = assetX * a
            cost = costFun(assetX)
            if sumBits == D:
                if cost < bestCost:  # collect the best(lowest) results results
                    bestPos = assetX
                    bestCost = cost
                    deg_m = [bestPos]
                elif cost == bestCost:
                    deg_m.append(assetX)

                if cost > worstCost:  # collect the worst(highest) results
                    worstPos = assetX
                    worstCost = cost
                    deg_g = [worstPos]
                elif cost == worstCost:
                    deg_g.append(assetX)

            if memory:  # record all qualified state
                if sumBits == D:
                    quRatio.append(assetX)
                    reNum.append(i)
                    res.append(cost)
    if verbose:
        print("The best position and cost is:", bestPos, bestCost)
        print("The best position in integer is:", bestPos / a)
        print("The worst position and cost is:", worstPos, worstCost)

    if memory:
        return quRatio, reNum, res, bestPos, bestCost, worstPos, worstCost, deg_m, deg_g
    else:
        return bestPos, bestCost


def get_RandSamps(
    n: int,
    costFun: Callable[..., float],
    l: int,
    u: int,
    amp: list[float],
    shots: int,
    total: int = False,
):
    """
    Randomly sample for shots

    n (int): the assets number
    l (int): the lower bound (included)
    u (int): the upper bound (included)
    total (int): fix the total number, default is False

    """
    sam = ClassRandSam(n=n, l=l, u=u, total=total)

    if total:
        myfun = sam.get_constrained_sum_sample_pos
    else:
        myfun = sam.get_bounded_sample_pos
    pos = [0 for _ in range(shots)]
    pos2 = [0.0 for _ in range(shots)]
    res = [0.0 for _ in range(shots)]
    for i in range(shots):
        pos[i] = myfun()
        pos2[i] = pos[i] * amp
        res[i] = costFun(pos2[i])

    return pos, pos2, res


def initParam(
    initBeta: Union[float, list],
    initGamma: Union[float, list],
    **kwargs: int,
):
    """
    generate the initial parameters,
    parameter start from the initial beta
    """
    if not isinstance(initGamma, list):
        initGammaList = initGamma * np.ones(shape=kwargs.get("p"))
    else:
        initGammaList = np.array(initGamma)

    if not isinstance(initBeta, list):
        initBetaList = initBeta * np.ones(shape=kwargs.get("p"))
    else:
        initBetaList = np.array(initBeta)

    initParams0 = np.concatenate((initBetaList, initGammaList))

    # Change the order
    # initParams = []
    # for i in range(2 * p):
    #     # print(i // 2 + (p * i) % 2)
    #     initParams.append(initParams0[i // 2 + (p * i) % 2])

    return initParams0


class cOptimization(object):
    """
    The class for classical optimization methods for VQE problem.

    """

    def __init__(self):
        """
        Class initialization.

        """
        pass

    def paramOptimize(
        self,
        objFun: Callable[..., float],
        method: str,
        initGamma: Union[float, list],
        initBeta: Union[float, list],
        p: int,
        numOuterIter: int,
        verbose: bool,
        tyPe: int,
        initTheta: float,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        """
        A universal optimization function. By selecting different methods we could optimize the objective function differently.

        Args:
            objFun: The objective function to be optimized.
            method: The classical optimization methods.
            initGamma: The initial value of gamma, by default pi.
            initBeta: The initial value of beta, by default pi/2.
            initTheta: Only works when xQAOA is used. By default None., for initial Optimization
            p: The number of layers of U_B and U_C in our problem.
            numOuterIter: The number of objective function evaluations, by default 1000. Notice that sometimes due to the implementation of the optimization methods, the actually number of objective function evaluations could exceed this number.
            tyPe: The type of circuit. It will affect the upper bound of betas.
            verbose: Whether print useful information or not.

        Return: The best possible pair of gamma and beta using the given method, as well as the corresponding optimal function value.
        """

        # Generate the initial parameters
        initParams = initParam(initGamma=initGamma, initBeta=initBeta, p=p)

        # Generate the bound for optimization
        lower, upper = boundPapa(p=p, tyPe=tyPe, initTheta=initTheta)

        if method == "compare":
            initParams = initParams[kwargs["cParas"]]
            lower = lower[kwargs["cParas"]]
            upper = upper[kwargs["cParas"]]

        # bounds = sp.optimize.Bounds(lb=lower, ub=upper, keep_feasible=True)
        bounds = list(zip(lower, upper))
        # More methods are yet to come.
        methods = {
            "Annealing": cOptimizer().annealing,
            "COBYLA": cOptimizer().COBYLA,
        }
        bestParams, bestEst = methods[method](
            objFun=objFun,
            initParam=initParams,
            lower=lower,
            upper=upper,
            bounds=bounds,
            numIter=numOuterIter,
            verbose=verbose,
            **kwargs,
        )

        return bestParams, bestEst
