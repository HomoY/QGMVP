# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""A Class for the quantum optimization part for our QAOA problem.
Now after so many versions, we need to come up with a cleaner one."""
__author__ = "HMY"
__date__ = "2021-09-18"

from typing import Tuple, Union, Callable, Any
import QGMVP.optimizer.cOpt as co
import numpy as np
import numba as nb
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from QGMVP.ansatz.Ansatz import simulate, EVQAA
from QGMVP.measure.stat import statz


class objFun(object):
    """
    Generate the objective function to be optimzied from the circuit
    """

    def __init__(
        self,
        circ: QuantumCircuit,
        simulator: Aer,
        numIter: int,
        costFun: Callable[..., float],
        eqv: EVQAA,
        noise: bool = False,
        consfilter: bool = True,
        para_verbose: bool = False,
        statMeas_verbose: bool = False,
        q_optimizer: str = "COBYLA",
        cParas: list = None,
        initParamsList: list = None,
    ):
        self.circ = circ
        self.simulator = simulator
        self.numIter = numIter
        self.costFun = costFun
        self.eqv = eqv
        self.noise = noise
        self.consfilter = consfilter
        self.para_verbose = para_verbose
        self.statMeas_verbose = statMeas_verbose
        self.q_optimizer = q_optimizer
        self.cParas = cParas
        self.initParamsList = initParamsList

    def paramsUpdate(self, params: np.ndarray):
        """Used to update params, if we need to use
        initParamsList[0...p-1 Beta from beta 0 to beta p-1, p+1 ... 2p-1 for gamma 0 to p]
        cParas[i,j] index the order initParams
        """

        if self.q_optimizer == "compare":
            paramsUpdated = self.initParamsList
            for i in range(len(self.cParas)):
                paramsUpdated[self.cParas[i]] = params[i]
            params = paramsUpdated

        return params

    def val(self, params: np.ndarray) -> float:
        """
        The objective function of quantum circuit given the parameters <gamma> and <beta>.

        params: An array of parameters, in the order, gamma, beta (, theta)
        circ (QuantumCircuit):

        Return: The function value.

        """
        assert self.circ is not None, "Set the circuit first!"
        params = self.paramsUpdate(params=params)
        # get the outcome of the simulations
        results = simulate(
            circ=self.circ,
            simulator=self.simulator,
            parameters=params,
            numIter=self.numIter,
        )
        objFunVal = statMeas(
            costFun=self.costFun,
            rc=results,
            numIter=self.numIter,
            eqv=self.eqv,
            noise=self.noise,
            consfilter=self.consfilter,
            memory=False,
            verbose=self.statMeas_verbose,
        )

        # print(objFunVal)
        if self.para_verbose == True:
            print(*params, sep=" ")
        # print("Objective function expected value：", objFunVal)

        return objFunVal


class qOptimization(object):
    """
    A class for quantum part of our QAOA problem. For example, the recursive QAOA.

    """

    def __init__(self, costObj: objFun, cOptimizer: co.cOptimization):
        """
        Class initialization.

        quadCost: The cost function object.
        cOptimizer: The optimizer object of the classical methods.
        tyPe: Type of circuit constraint. 1 for soft, 2 for hard.
        struct: Structure of our QAOA scheme.
        num: The number of total available assets in our problem.
        foTerm: First order term.
        soTerm: Second order term.
        p: The interation of the Ansatz circuit
        initPos: The initial position of assets. Only available for hard constraint.
        simult: Whether using simultaneous circuit or not.

        """
        self.costObj = costObj
        self.cOptimizer = cOptimizer

    def opt(
        self,
        method: str = "COBYLA",
        initGamma: Union[float, list] = 0.1,
        initBeta: Union[float, list] = 0.1,
        p: int = 1,
        numOuterIter: int = 1000,
        verbose: bool = True,
        tyPe: int = 1,
        initTheta: float = None,
        optTar: str = "mean",
        **kwargs,
    ) -> Tuple[np.array, float]:
        """
        The QAOA procedure.

        The parameters are just as that used in paramOptimization in cOptimization.

        """
        if optTar == "mean":
            myfun = self.costObj.val

        return self.cOptimizer.paramOptimize(
            myfun,
            method,
            initGamma,
            initBeta,
            p,
            numOuterIter,
            verbose,
            tyPe,
            initTheta,
            **kwargs,
        )


@nb.jit(nopython=True, nogil=True)
def estMean(costArray: np.ndarray, cntArray: np.ndarray, numIter: int) -> float:
    """
    Estimate the mean of the Hamiltonian, i.e., E{< Psi|C|Psi >}.

    resArray: A two dimensional array containing the simulation results, [[binNum, count]]
    costInfo: The memorized cost function. Just to save time if possible.
    numIter: The total number of iterations.

    Return: The estimated mean of the Hamiltonian, given parameters gamma and beta.

    """
    m = costArray @ cntArray / numIter

    return m


# @spendtime
def statMeas(
    costFun: Callable[..., float],
    rc: Union[dict, Any],
    eqv: EVQAA,
    numIter: int = None,
    noise: bool = False,
    memory: bool = False,
    verbose: int = 0,
    method: str = "mean",
    consfilter: bool = True,
    **kwargs,
):
    """
    Get the statsitc analysis after measurements with respect to your function
    we put the string length check in the front in case we find nothing in it
    costFun: the fucntion you want to use
    measure is the outputs from get_counts of measurement
    eqv: the setting of qubits

    return:
    xs: measured qubits projection to position/ratioof investment x
    cs: the cost value to each measurement results
    meanVal: the mean values of the experiment
    minVal: the min value of measurement and the corresponding measurement
    maxVal: the max value of measurement
    noise: if noise
    variance: to get the variance of the measurement

    """
    statzClas = statz(
        costFun=costFun,
        rc=rc,
        eqv=eqv,
        noise=noise,
        memory=memory,
        verbose=verbose,
        numIter=numIter,
        consfilter=consfilter,
        **kwargs,
    )
    # if noise we filter out the bad results, but I might change this later
    if memory == False and method == "mean":
        return statzClas.fast_get_mean()
    elif memory == "measuredict":
        return statzClas.get_measuredict()
