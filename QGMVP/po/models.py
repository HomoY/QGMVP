# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Generate models by random method or the set files 
"""
__author__ = "HMY"
__date__ = "2024-Aug-22"


import pickle
import numpy as np
from random import random, sample, randint, uniform
import os
import igraph as ig
from typing import Union


class ClassRandSam(object):
    """Integrate the random sampling function
    n: the number of assets
    l: the lower bound
    n: the upper bound
    total: the total sum of the assets only for the constrained sum sample
    """

    def __init__(self, n: int, l: int = 0, u: int = 1, total=None) -> None:
        self.n = n
        self.l = l
        self.u = u
        self.total = total

    def get_bounded_sample_pos(self):
        return bounded_sample_pos(n=self.n, l=self.l, u=self.u)

    def get_constrained_sum_sample_pos(self):
        return constrained_sum_sample_pos(n=self.n, total=self.total)


def bounded_sample_pos(n: int, l: int = 0, u: int = 1):
    """
    Return a randomly chosen list of n positive integers with upper and lower bound.

    Args:
        n (int): the asset's number
        l (int): the lower bound (included)
        u (int): the upper bound (included)

    """

    return [randint(l, u) for _ in range(n)]


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def constrained_sum_sample_nonneg(n, total):
    """Return a randomly chosen list of n nonnegative integers summing to total.
    Each such list is equally likely to occur."""

    return [x - 1 for x in constrained_sum_sample_pos(n, total + n)]


def rg(n: int, alpha: float = False):
    """_summary_

    Args:
        n (int): asset number
        N (int): qubit number
        alpha (int): amplification
    """
    lamb = random()  # the balance
    sigma = np.array([[uniform(-1, 1) for _ in range(n)] for _ in range(n)])
    sigma = 0.5 * (sigma + sigma.T)  # symetric sigma
    mu = np.array(
        [random() for _ in range(n)]
    )  # returns, we here not consider when covariance is large, and mu is small situation but just add everything into
    if alpha:
        Ialpha = int(1 // alpha)  # Inverse of alpha
        x_testRN = constrained_sum_sample_nonneg(n, Ialpha)
        x_test = np.array([x_testRN[i] / Ialpha for i in range(n)])
        return sigma, mu, lamb, x_testRN, x_test

    return sigma, mu, lamb


def modelGene(
    n: int,
    rc: bool = False,
    alpha: int = None,
    method: str = "random",
    set: str = None,
    test_sets: str = False,
    verbose: bool = True,
    amplify: float = 1.0,
    *args,
    **kwargs,
):
    """
    The library is used to generate target problems' parameters,
        Args:
        n (int): asset number ref parameters.py
        rc (bool): random configuration, if True, generate a random parameter set; False, find the corresponding parameters from expParam file; if not find will prompt error
        test (bool) default True: if generate a test string, True, generate; False, not generates
        alpha: (float) default 'None', ref in ansatz/Ansatz/EVQAA
        Returns:
        sigma (np.ndarray): n*n float, symmetric covariance matrix for the financial model
        mu (np.ndarray): 1*n float, expected return vector.
        
        method (str): the method to generate the parameters, default is random
            method = 'random': generate the parameters randomly
            method = 'set': generate the parameters from the set files
            

        x_test (np.ndarray): 1*n float, a normalized ratio under sum constraint and with accuracy under the current number of qubits size
        x_testRN (np.ndarray): 1*n int, same as x_test but without renormalization with int inputs
        lamb (float): The parameter describing the risk aversion ratio of the investor, a larger lamb means the investor counts more about expected returns
    Pointed to the files to read the appropriate files

    """

    def fun_set(n, set, verbose):
        try:
            with open(
                f"{os.getcwd()}/po/expParam/" + set + ".pkl", "rb"
            ) as f:  # Python 3: open(..., 'rb')
                sigma, mu = pickle.load(f)

                if len(sigma) != n:
                    raise Exception(
                        "The number of asset is not the same as the imported set file"
                    )
                if verbose:
                    print("Covariance matrix: ")
                    print(sigma)
                    print()
                    print("Expected return: ")
                    print(mu)
        except:
            raise Exception("Sorry, we have yet not include the example parameters ")

        return np.array(sigma), np.array(mu)

    def fun_test_set(test_sets):
        """Generate the test set as for the parameters"""
        try:
            with open(
                f"{os.getcwd()}/po/expParam/" + test_sets + ".pkl", "rb"
            ) as f:  # Python 3: open(..., 'rb')
                _, data = pickle.load(f)

            sigmas = []
            mus = []
            for _, value in enumerate(data):
                sigmas.append(data[value]["Qk_U"])
                mus.append(data[value]["qk_U"])

            return sigmas, mus
        except:
            raise Exception("Sorry, we have yet not include the example parameters ")

    if verbose:
        print("\nInitialisation for classical objective function...\n")

    if method == None:
        raise Exception("No method is appointed!")
    elif method == "random":
        if verbose:
            print("The object configuration is generated randomly")
        return rg(n=n, alpha=alpha)
    elif method == "set":
        return fun_set(n, set, verbose)
    elif method == "test_sets":
        return fun_test_set(test_sets)
