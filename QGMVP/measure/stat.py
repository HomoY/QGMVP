# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""a postprocessing for the measurements"""
__author__ = "HMY"
__date__ = "2023-Jul-13"
from typing import Tuple, Union, Callable, Any
from QGMVP.encode import evqa, enco
import numpy as np
import numba as nb
import math
from collections import OrderedDict


class statz(enco):
    """
    A class for all statistic method for the post data of quantum circuit if measured in z basis

    """

    def __init__(
        self,
        costFun: Callable[..., float],
        rc: Union[dict, Any],
        eqv: evqa,
        noise: bool = False,
        memory: bool = False,
        verbose: list = False,
        consfilter: bool = True,
        **kwargs,
    ) -> None:
        self.costFun = costFun
        self.rc = rc

        if "numIter" in kwargs:
            self.numIter = kwargs["numIter"]
        else:
            self.numIter = sum(rc.values())

        self.length = len(
            rc
        )  # this is only for the unique measurements length if noise, we will check the length again in case we deal with the situation when no measurements left

        if self.length == 0:
            raise Exception("Check the meaurement, nothing in it!")

        self.eqv = eqv
        self.noise = noise
        self.consfilter = consfilter
        self.memory = memory
        self.verbose = verbose
        self.kwargs = kwargs
        if not self.noise:
            self.myfun = bin2x
        elif self.noise and self.consfilter:
            self.myfun = BudgetConstraint
        elif self.noise and not self.consfilter:
            self.myfun = bin2x

    def get_string(self):
        """Get the string cost and corresponded frequecy array"""
        self.costArray = np.zeros(shape=self.length, dtype=np.float64)
        self.cntArray = np.fromiter(self.rc.values(), dtype=np.float64)

        for i, j in enumerate(self.rc):
            self.costArray[i] = self.costFun(
                self.myfun(
                    s=j,
                    n=self.eqv.n,
                    posQ=self.eqv.posQ,
                    posI=self.eqv.posI,
                    encoN=self.eqv.encoN,
                )
            )

    def fast_get_mean(self):
        """Get the mean in the fatest way, used for optimisation"""
        if self.noise == False:
            self.get_string()
            meanVal = estMean(
                costArray=self.costArray,
                cntArray=self.cntArray,
                numIter=sum(self.cntArray),
            )
        else:
            meanVal = self.memoryFunFilter(noiseVerbose=True)[5]
        self.info(input=meanVal)
        return meanVal

    def fast_get_mean_noise(self):
        """Get the mean in the fatest way, used for optimisation for the noise situation"""
        meanVal = self.memoryFun()[5]
        self.info(input=meanVal)

        return meanVal

    def fast_get_var(self):
        """Get the variance in the fatest way, used for optimisation"""
        self.get_string()
        varVal = estVar(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        self.info(input=varVal)

        return varVal

    def mean_and_var(self):
        """Get the mean and variance in the fatest way and return both"""
        self.get_string()
        meanVal = estMean(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        varVal = estVar(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        self.info(input=[meanVal, varVal])

        return meanVal, varVal

    def mean_and_var_return_mean(self):
        """Get the mean and variance in the fatest way, but only return mean"""
        self.get_string()
        meanVal = estMean(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        varVal = estVar(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        self.info(input=[meanVal, varVal])

        return meanVal

    def mean_and_var_return_var(self):
        """Get the mean and variance in the fatest way, but only return variance"""
        self.get_string()
        meanVal = estMean(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        varVal = estVar(
            costArray=self.costArray, cntArray=self.cntArray, numIter=sum(self.cntArray)
        )
        self.info(input=[meanVal, varVal])

        return varVal

    def get_percent(self, percent: float = 0.05):
        """This is used for getting percentage rate(if shots)/probability (if density matrix) with the lowest energy

        percent (float): the percentage of the best you may want to get

        return:
                EnergyDict (dict): {f(x): Prob/cntArray} from f(x) small to large and additive Prob is lower than set percent
                flag: the additive probability of the EnergyDict
        """
        _outputs = self.memoryFun(sort="cost")
        # numShotsTotal is estamate how many shots in total,
        # numShots is the shots you need to look into
        # The below function will do accumulation expectation of the value
        numShotsTotal = sum(_outputs[3].values())
        numShots = numShotsTotal * percent
        flag = 0
        flag2 = 0
        EnergyDict = {}
        for _, value in enumerate(_outputs[3]):
            flag2 = _outputs[3][value]

            if flag + flag2 >= numShots:
                EnergyDict[value] = numShots - flag
                break
            else:
                flag += flag2
                EnergyDict[value] = flag2

        return (EnergyDict, flag)

    def get_lower(self, cuteng: float):
        """This is used for getting accumulated percentage rate(if shots)/probability (if density matrix) with cut of enengy preset

        cuteng (float): the cut energy, the system will remember the percentage rate of energy below this cut energy
        """
        _outputs = self.memoryFun(sort="cost")

        flag = 0
        for _, value in enumerate(_outputs[3]):
            if value <= cuteng:
                flag += _outputs[3][value]

        return flag

    def get_bins(
        self, binn: int, l: float, u: float, boundary: int = 1, cdf: bool = False
    ):
        """
        This is used for getting bins of percentage rate(if shots)/probability (if density matrix) with cut of enengy preset

        Args:
            binn (int): bin number you want to divide >= 2 else it would makes no sense to use this function
            l (float): the lower of the bins
            u (float): the upper of the bins
            boundary (int): different boundary condtion mode, includeness of the right or left bounndary. Defaults to 1: [,). [,) ,..., [,] If 0: [,], (,],(,],(,]
            cdf (bool): if True, vals = cdf else vals = pdf

        """
        _outputs = self.memoryFun()

        gap = (u - l) / binn
        bins = [i for i in range(binn + 1)]
        data = dict.fromkeys(i for i in range(binn))

        if boundary == 1:
            for index, value in enumerate(_outputs[3]):
                bins[math.floor((index - l) / gap)] = value
            bins[1] += data[0]
            if cdf == False:
                data.update((k, bins[i + 1]) for i, k in enumerate(data))
            else:
                data.update((k, bins[0 : i + 1]) for i, k in enumerate(data))
        elif boundary == 2:
            for index, value in enumerate(_outputs[3]):
                bins[math.ceil((index - l) / gap)] = value
                bins[binn - 1] += data[binn]
            if cdf == False:
                data.update((k, bins[i]) for i, k in enumerate(data))
            else:
                data.update((k, bins[0:i]) for i, k in enumerate(data))

        return data

    def get_mode_with_percent(self, top: int = 1):
        """This is used for getting most percentage of the most appeared states

        top (float): the top how many best
        """
        _outputs = self.memoryFun(sort="frequency")
        flag = 0
        newlist = {}
        for index, value in enumerate(_outputs[3]):
            if index < top:
                flag += _outputs[3][value]
            newlist.update({value: _outputs[3][value]})
        return flag, newlist

    def get_list(self, findList: list, spc: str = "c"):
        """This is for getting a list of position's amplitudes rate(if shots)/probability (if density matrix)

        spc (str): "s" means string for example "10011"
                   "p" means positions for example "0.333, 0.6666"
                   "c" means cost value
        """
        flag = 0
        _outputs = self.memoryFun()
        lenList = len(findList)
        if spc == "p":
            for i in range(lenList):
                newstr = list(_outputs[0].keys())[
                    list(_outputs[0].values()).index(findList[i])
                ]  # get the string out from the positions

                newcost = _outputs[1][newstr]  # convert the string to the cost
                flag += newcost  # get the frequency out from the cost
        if spc == "c":
            for i in range(lenList):
                try:
                    newcost = _outputs[3][findList[i]]
                    flag += newcost
                except:
                    print("Not find", findList[i], "in the list")
                    flag += 0.0
        elif spc == "p":
            for i in range(lenList):
                flag += _outputs[3][findList[i]]

        return flag

    def get_seecirc(self):
        """an api for the seecirc include all information"""
        return self.memoryFun()

    def get_measuredict(self, _sort=True, keyhead: str = "x"):
        """an api for the post measure sorted dict include all needed information

        keyhead (str): the head of the key, for example, "bit" means the key is "bit0", "bit1", "bit2"... if "x", will change the key as tuple("x")

        measuredict: dict, a measure dictionary
        optdict: optimial measured dictionary
        """
        _outputs = self.memoryFun(sort=False)

        measuredict = {}

        for key, value in enumerate(_outputs[0]):
            measuredict[value] = {
                "x": _outputs[0][value],
                "cost": _outputs[1][value],
                "freq/prob": _outputs[2][key],
            }

        def sort_dict_by_val(input_dict, _sort):
            if _sort:
                # Sort the dictionary by the 'val' key in the nested dictionaries
                sorted_items = sorted(
                    input_dict.items(), key=lambda item: item[1]["cost"]
                )
                # Convert the sorted items back to a dictionary
                sorted_dict = {k: v for k, v in sorted_items}
                return sorted_dict
            else:
                return input_dict

        def transform_dict(input_dict, keyhead):
            if keyhead == "bit":
                return input_dict
            elif keyhead == "x":
                transformed_dict = {}
                for key, value in input_dict.items():
                    x_value = tuple(value[keyhead])
                    transformed_dict[x_value] = {
                        "bit": key,
                        "cost": value["cost"],
                        "freq/prob": value["freq/prob"],
                    }
                return transformed_dict

        measuredict = transform_dict(
            sort_dict_by_val(measuredict, _sort=_sort), keyhead=keyhead
        )

        optdict = {
            "meanVal": _outputs[5],
            "varVal": _outputs[6],
            "minDict": _outputs[7],
            "errP": _outputs[8],
            "errL": _outputs[9],
        }

        return measuredict, optdict

    def get_fun_prob(self, noise=False):
        """Get fun(x): probability"""
        _, _, _, cp, _, _ = self.meaSxScProb(noise=noise)
        return cp

    def scmmdis(self):
        """designed for the cost distribution, sx, cs, meanVal. minDict"""
        _outputs = self.memoryFun()
        return _outputs[0], _outputs[1], _outputs[5], _outputs[7]

    def scmm(self):
        """designed for the sx, cs, meanVal. minDict"""
        _outputs = self.memoryFun()
        return _outputs[0], _outputs[1], _outputs[5], _outputs[7]

    def smvm(self):
        """designed for the sx, mean, variance, minDict"""
        _outputs = self.memoryFun(noiseVerbose=True)
        return _outputs[0], _outputs[5], _outputs[6], _outputs[7]

    def memoryFun(
        self,
        sort=False,
        er=1e-15,
        noiseVerbose=False,
    ) -> Tuple[dict, ...]:
        """return all memoried results deserved to be used


        Returns:
            Tuple[dict, ...]:
            mesdicts (dict): bit {string: "x": x, "cost": cost, "freq/prob": prob}
            sx is sting to x list, warming: not used anymore,
            sc is string to cost,  warming: not used anymore,
            cntArray is the frequency/probability,  warming: not used anymore,
            sp : {f(x): Prob/cntArray} for each x
            costArray: f(x),
            meanVal: mean value,
            minDict: the minima diction
        """
        sx, sc, sp, cp, errP, errL = self.meaSxScProb(
            noise=self.noise,
            sort=sort,
            er=er,
            noiseVerbose=noiseVerbose,
        )

        self.cntArray = np.fromiter(sp.values(), dtype=np.float64)  # probabilities
        numIter = sum(self.cntArray)
        self.costArray = np.array(
            list(sc.values()), dtype=np.float64
        )  # cost values arrays

        meanVal = estMean(
            costArray=self.costArray, cntArray=self.cntArray, numIter=numIter
        )
        varVal = estVar(
            costArray=self.costArray, cntArray=self.cntArray, numIter=numIter
        )

        minDict = {min(sc, key=sc.get): min(self.costArray)}
        mesdicts = {}
        return (
            sx,
            sc,
            self.cntArray,
            sp,
            self.costArray,
            meanVal,
            varVal,
            minDict,
            errP,
            errL,
        )

    def memoryFunFilter(
        self,
        sort=False,
        er=1e-15,
        noiseVerbose=False,
    ) -> Tuple[dict, ...]:
        """return all memoried results deserved to be used


        Returns:
            Tuple[dict, ...]: sx is sting to x list, sc is string to cost, cntArray is the probability, cp : {f(x): Prob/cntArray}
            costArray: f(x), meanVal: mean value, minDict: the minima diction
        """
        if not self.consfilter:
            sx, sc, sp, cp, errP, errL = self.meaSxScProb_nofilter(
                noise=self.noise,
                sort=sort,
                noiseVerbose=noiseVerbose,
                er=er,
            )
        else:
            sx, sc, sp, cp, errP, errL = self.meaSxScProb(
                noise=self.noise,
                sort=sort,
                er=er,
                noiseVerbose=noiseVerbose,
            )

        self.cntArray = np.fromiter(cp.values(), dtype=np.float64)  # probabilities
        numIter = sum(self.cntArray)
        self.costArray = np.array(
            list(sc.values()), dtype=np.float64
        )  # cost values arrays

        costArrayEst = np.array(list(cp.keys()), dtype=np.float64)  # cost values arrays

        meanVal = estMean(
            costArray=costArrayEst, cntArray=self.cntArray, numIter=numIter
        )
        varVal = estVar(costArray=costArrayEst, cntArray=self.cntArray, numIter=numIter)

        minDict = {min(sc, key=sc.get): min(self.costArray)}

        return (
            sx,
            sc,
            self.cntArray,
            cp,
            self.costArray,
            meanVal,
            varVal,
            minDict,
            errP,
            errL,
        )

    def meaSxScProb(
        self,
        noise: bool = False,
        sort: bool = False,
        noiseVerbose: bool = False,
        er=1e-15,
        combdict: bool = False,
    ) -> dict:
        """
        A dictional transfer from a measure get_count() of qiskit to another diction map all measurements to x, and record the frequency
        This is a post processing of the string by string, it is slow but works

        self.rc: measured results from get_count()
        er: the error for comparing to rules out the one with little probabilities but this is not for the shots measure because shots can only be integer
            The error occurs because of the qiskit itself, qiskit will generate phase to the circuit because it only support the single and float accuracy when they perform
            StateVector for the probabilty or phase, if float(16), it confues people if we get a proper phase or just the noise when we have 16 qubits system and apply a Hadarmard transformation

        noiseVerbose: if True, print the error ratio (errL - errP) / errL

        need self.eqv's parameter including: n, posQ, posI, encoN: should go to the def in EQVAA
        memory (bool): if True, return the diction of measure -> x; False return xArray and cntArray(for optimizing)
        combdict (bool): if True, combine to the measdicts as bit {string: "x": x, "cost": cost, "freq/prob": prob}

        Return:
            Tuple[dict, ...]: sx is sting to x list, sc is string to cost, cntArray is the probability, cp : {f(x): Prob/cntArray}
            errP
            err
        """
        sx = {}  # a dictionary to store dictionary from string to x value set
        errP = 0  # error percentage
        errL = []  # error list
        sc = {}  # string: cost
        cp = {}  # sc after removing replications postprocessing
        sp = {}  # string to probability

        if noise:
            myfun = BudgetConstraint
        else:
            myfun = bin2x

        for k, i in enumerate(self.rc):
            if self.rc[i] <= er:
                continue
            c = myfun(
                s=i,
                n=self.eqv.n,
                posQ=self.eqv.posQ,
                posI=self.eqv.posI,
                encoN=self.eqv.encoN,
            )  # transfer the binary measure string to x
            if c.any():
                sx[i] = c
                _fun = self.costFun(sx[i])
                sc[i] = _fun
                sp[i] = self.rc[i]

                if _fun in cp.keys():
                    cp[_fun] += self.rc[i]
                else:
                    cp[_fun] = self.rc[i]
            else:
                errL.append(i)
                errP = errP + self.rc[i]

        # Sometimes, you will find the whole circuit could be empty, and then you would need to check out your outputs and may lead to the crash of your circuit optimisation
        length_post = len(sx)
        if length_post == 0:
            raise Exception("Check the meaurement, nothing in it!")

        if sort == "cost":
            cp = dict(sorted(cp.items()))
        elif sort == "frequency":
            cp = dict(sorted(cp.items(), key=lambda x: x[1], reverse=True))

        if noiseVerbose == True:
            print((self.numIter - errP) / self.numIter, end=" ")

        return sx, sc, sp, cp, errP, errL

    def meaSxScProb_nofilter(
        self,
        noise: bool = False,
        sort: bool = False,
        noiseVerbose: bool = False,
        er=1e-15,
    ) -> dict:
        """
        A dictional transfer from a measure get_count() of qiskit to another diction map all measurements to x, and record the frequency
        This is a post processing of the string by string, it is slow but works

        self.rc: measured results from get_count()
        er: the error for comparing to rules out the one with little probabilities but this is not for the shots measure because shots can only be integer
            The error occurs because of the qiskit itself, qiskit will generate phase to the circuit because it only support the single and float accuracy when they perform
            StateVector for the probabilty or phase, if float(16), it confues people if we get a proper phase or just the noise when we have 16 qubits system and apply a Hadarmard transformation

        noiseVerbose: if True, print the error ratio (errL - errP) / errL

        need self.eqv's parameter including: n, posQ, posI, encoN: should go to the def in EQVAA
        memory (bool): if True, return the diction of measure -> x; False return xArray and cntArray(for optimizing)

        Return:
            Tuple[dict, ...]: sx is sting to x list, sc is string to cost, cntArray is the probability, cp : {f(x): Prob/cntArray}
            errP
            err
        """
        sx = {}  # a dictionary to store dictionary from string to x value set
        errP = 0  # error percentage
        errL = []  # error list
        sc = {}  # string: cost
        cp = {}  # sc after removing replications postprocessing
        sp = {}  # string to probability

        myfun = BudgetConstraintUnfil

        for k, i in enumerate(self.rc):
            if self.rc[i] <= er:
                continue
            c1, c2 = myfun(
                s=i,
                n=self.eqv.n,
                posQ=self.eqv.posQ,
                posI=self.eqv.posI,
                encoN=self.eqv.encoN,
            )  # transfer the binary measure string to x
            sx[i] = c2
            _fun = self.costFun(sx[i])
            sc[i] = _fun
            sp[i] = self.rc[i]

            # remove replications
            if _fun in cp.keys():
                cp[_fun] += self.rc[i]
            else:
                cp[_fun] = self.rc[i]

            if not c1:
                errL.append(i)
                errP = errP + self.rc[i]

        if noiseVerbose == True:
            print((self.numIter - errP) / self.numIter, end=" ")

        return sx, sc, sp, cp, errP, errL

    def info(self, input):
        """Get the verbose of returns"""
        if self.verbose == 1:
            print(input)
        elif self.verbose == 2:
            print(input, min(self.costArray), max(self.costArray))
        elif self.verbose == 3:
            mp = self.costArray.argmin()
            Mp = self.costArray.argmax()
            str1 = np.array(list(self.rc.keys()))
            print(input, self.costArray[mp], self.costArray[Mp], str1[mp], str1[Mp])
        elif self.verbose == 4:  # designed only for memory case
            pass


def statc(
    cntArray: np.ndarray,
    costArray: np.ndarray,
):
    """get the statistc according to the cost value

    cntArray (np.ndarray): possibilities of measurement
    costArray (np.ndarray): cost strings
    """
    pass


def bin2x(
    s: str,
    n: int,
    posQ: list,
    posI: list,
    encoN: list,
):
    """
    [Transform binary to asset position x]

    s: The measurement binary string
    loc: an array calls the coefficient point to each qubit measurement,
    n: the size of asset
    posQ:
    encoN

    Return:
    x: the measured qubit value represent,
    v: each qubit represent the binary value

    Example:
    s = "01110100001000"
    loc = array([[1, 2, 1], # the first lines means from left to right the first qubit's coefficent is 1, the second and third qubits' coefficient is 2, the fourth qubits coefficient is 4, and total qubits to represent the first asset is 4
        [1, 2, 0],
        [1, 2, 0],
        [1, 1, 0],
        [2, 0, 0]])

    x = array([8, 2, 0, 1, 0])
    v = array([1, 2, 2, 4, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1])
    """
    x = [0] * n
    for i in range(n):
        for j in posI[i]:
            x[i] += encoN[i][j] * int(s[posQ[i][j]])

    return np.array(x)


def BudgetConstraint(
    s: str,
    n: int,
    posQ: list,
    posI: list,
    encoN: list,
):
    """Check one str if qualified for the str

    Args:
        str (_type_): _description_
    """
    x = bin2x(
        s=s,
        n=n,
        posQ=posQ,
        posI=posI,
        encoN=encoN,
    )  # transfer the binary measure string to x
    if sum(x) == 1:
        return x
    else:
        return np.array(False)


def BudgetConstraintUnfil(
    s: str,
    n: int,
    posQ: list,
    posI: list,
    encoN: list,
):
    """Check one str if qualified for the str

    Args:
        str (_type_): _description_
    """
    x = bin2x(
        s=s,
        n=n,
        posQ=posQ,
        posI=posI,
        encoN=encoN,
    )  # transfer the binary measure string to x
    if sum(x) == 1:
        return True, x
    else:
        return False, x


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


@nb.jit(nopython=True, nogil=True)
def estVar(costArray: np.ndarray, cntArray: np.ndarray, numIter: int) -> float:
    """
    Estimate the variance of the Hamiltonian, i.e., E{< Psi|C^2|Psi >}.

    resArray: A two dimensional array containing the simulation results, [[binNum, count]]
    costInfo: The memorized cost function. Just to save time if possible.
    numIter: The total number of iterations.

    Return: The estimated variance of the Hamiltonian, given parameters gamma and beta.

    """
    m = ((costArray**2 @ cntArray) / numIter) - (costArray @ cntArray / numIter) ** 2
    ((costArray**2 @ cntArray) / numIter) - (costArray @ cntArray / numIter) ** 2

    return m


class NoiseFilter(object):
    """This one is for noise post-process"""

    def __init__(self, eqv: evqa):
        self.eqv = eqv

    def RatioDict(self, ratio: dict):
        """{ratio:}"""
        pass

    def StrDicts(self, rc: dict):
        """Noise {str:probability} -> Constrainted {str:probability}, for checking if measured result can be used for filtering

        Args:
            rc (dict): Noise {str:probability} from measurement
        """
        for i, j in enumerate(rc):
            if (
                BudgetConstraint(
                    s=j,
                    n=self.eqv.n,
                    posQ=self.eqv.posQ,
                    posI=self.eqv.posI,
                    encoN=self.eqv.encoN,
                )
                == False
            ):
                del rc[j]

        return rc
