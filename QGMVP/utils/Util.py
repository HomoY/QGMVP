# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""A file of utilities"""
__author__ = "HMY"
__date__ = "2025-07-07"

from typing import Callable, Tuple, Union, Any
import pandas as pd
import numpy as np
import os
import pickle


def dataDfDict(name: list, data: list, *args, **kwargs):
    """convert a dict of data to the pandas.DataFrame struture

    Returns:
        name (list): the describe of the data
        data (list): the list of data
    """
    c = {}
    for i in name:
        c[i] = []
    for i in range(len(data)):
        for j in range(len(name)):
            c[name[j]].append(data[i][name[j]])

    return pd.DataFrame(c)


def pickleUpt(path: None, *args):
    """
    The function will update a parameter to a file
    The function first load the file of path pointed, and then check all the args want to add to append to the pkl path
    """
    try:
        with open(path, "rb") as f:  # Python 3: open(..., 'rb')
            c = pickle.load(f)
    except:
        c = []
        print("Error, no files in path, will create one!")

    for i in range(len(args)):
        c.append(args[i])

    with open(path, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(c, f)

    return print("Updated")


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


def boundPapa(p: int, tyPe: int, **kwargs):
    """
    Define the bound for the parameters
    tyPe (int): used for model type, may discard if no need
    initTheta (bool): for xQAOA bound

    return: lower and higher bound (The first p elements is for the mixing operator, and the rest is for the cost operator)
    """
    lower = np.zeros(shape=2 * p)
    upper = np.append(2 * np.pi * np.ones(shape=p), 2 * np.pi / np.ones(shape=p) / tyPe)

    # For xQAOA
    if "initTheta" in kwargs:
        if kwargs["initTheta"] is not None:
            initParams = np.append(initParams, kwargs["initTheta"])
            lower = np.append(lower, -1)
            upper = np.append(upper, 1)

    return lower, upper


def initGene(
    initGamma: Union[list, np.ndarray],
    initBeta: Union[list, np.ndarray],
    p: int,
    tyPe: int = 1,
    **kwargs,
):
    """This function is used for generating an initGamma list
    init (list): a list for gamma and beta
    p (int): the length of prolonged size

    return
        initGammaGene: a list of initial Gamma array
        initBetaGene: a list of initial Beta array
    """
    lower, upper = boundPapa(p=p, tyPe=tyPe, **kwargs)

    initGamma = [np.float64(x) for x in initGamma]
    initBeta = [np.float64(x) for x in initBeta]
    initGammaGene = list(initGamma) + list(
        (np.random.rand(p - len(initGamma))) * (upper[p] - lower[p]) + lower[p]
    )
    initBetaGene = list(initBeta) + list(
        (np.random.rand(p - len(initBeta))) * (upper[0] - lower[0]) + lower[0]
    )
    if "bound_verbose" not in kwargs:
        return initGammaGene, initBetaGene
    else:
        if kwargs["bound_verbose"] == True:
            return initGammaGene, initBetaGene, lower, upper
        else:
            return initGammaGene, initBetaGene


def readMulRun(filename):
    # mr = []  # MulRun solution saver
    # read the MulRun according to the files
    # The below code will count how many MulRun I generate in my running of progam and count them
    flag = 0
    for item in os.listdir(filename):
        if "MulRun" in item:
            flag += 1

    mr = [None for _ in range(flag)]  # MulRun saver

    # The below program will check if the MulRun is in item and put them in the correct position of tmpmr list
    for item in os.listdir(filename):
        if "MulRun" in item:
            j = int(
                item[6:-4]
            )  # if I change the name rules of dump files I would also need to modify this lines, this one is for MulRunint.pkl
            with open(filename + "/" + item, "rb") as f:
                mr[j] = pickle.load(f)

    return mr


def readMulRunIndex(filename, index):
    """Get the MulRunin the filename folders according to the specific index

    Args:
        filename (_type_): The folders of MulRun
        index (_type_): the number of the MulRun want to save

    Returns:
        _type_: _description_

    Exceptions:
        Exception: The file has exists! if exist file in initial
    """

    # The below program will check if the MulRun is in item and import the result
    with open(filename + "/" + "MulRun" + str(index) + ".pkl", "rb") as f:
        mr = pickle.load(f)

    return mr
