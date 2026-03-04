# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""A cost ansatz generator for Efficient Variational quantum Approximation algorithom"""
__author__ = "HMY"
__date__ = "2023-01-24"

from typing import Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from QGMVP.ansatz.Ansatz import EVQAA
from numpy.typing import NDArray


class costCoe(object):
    """
    Initialized to generate the cost operator coefficients

    """

    def __init__(self, eqv: EVQAA, qk: NDArray, Qk: NDArray, obj_model="gmvp") -> None:
        """
        cost (costsfun): get the information about the cost functions
        eqv (EVQAA): parameters about quantum sets
        qk (NDArrayy): single interation coefficient
        Qk (NDArray): covariance interation coefficient
        """
        self.eqv = eqv
        self.qk = qk
        self.Qk = Qk
        if obj_model == "gmvp":
            self.qke, self.Qke = self.extqkQk(check=False)

    def extqkQk(
        self,
        check: bool = False,
    ):
        """This function is used for generating the first and second order coefficients
            the measured results are -11-11-11 such configuration
        Args:
            check (bool), check if the output is correct or not， generate constant, for test use
        return:
        qke (np.ndarray): the expanded single term
        Qke (np.ndarray): the expanded interacted term
        c (optional)
        """

        # suggest the tensor expansion is the same for all aseet
        a = 1 / (2 ** self.eqv.allo[0] - 1)
        outPV = np.array(
            [(2**i) / 2 * a for i in range(self.eqv.allo[0])]
        )  # generate outer product
        outPM = np.einsum("i,j", outPV, outPV)  # generate the outer product matrix

        Qke = np.kron(self.Qk, outPM)  # generate the Qk expanded
        qke = -np.einsum("ij->i", Qke) - np.kron(
            self.qk, outPV
        )  # generate the qk expanded

        if check:
            c = (
                np.einsum("i->", np.kron(self.qk, outPV)) + np.einsum("ij->", Qke) / 2
            )  # the constant part only for check if the result is correct, only on identity
            return qke, Qke, c
        else:
            return qke, Qke


class exp2Z(QuantumCircuit):
    """Generate a quantum circuit of exp(-i g Zi Zj),
        where i, j are the connected two qubits name

    g is the parametered angle before Zi Zj operators
    theta is the extra angle for the circuit

    """

    def __init__(
        self,
        g: Union[float, Parameter, str],
    ):
        if isinstance(g, str):
            g = Parameter(g)

        qn = QuantumRegister(2)
        qc = QuantumCircuit(qn)

        qc.cx(qn[0], qn[1])
        qc.rz(2 * g, qn[1])
        qc.cx(qn[0], qn[1])

        super().__init__(*qc.qregs, name="e(-gZij)")
        self.compose(qc, qubits=self.qubits, inplace=True)


class D123(QuantumCircuit):
    """
    Generate the D(1), D(2), D(3) layer for each cycle

    Args:
        N (int): the number of qubits of circuit
        D (int): is the layer we are looked into, start from 1 to N
        Qke (np.ndarray): two dimension matrix with N*N size
        g (float): the angle for the cost operators
    (we don't include the diagonal terms, because they are constant in the end)
    """

    def __init__(
        self,
        N: int,
        D: int,
        Qke: np.ndarray,
        g: Union[float, Parameter, str],
    ):
        if isinstance(g, str):
            g = Parameter(g)
        self.N = N
        self.D = D
        self.Qke = Qke
        self.g = g
        qn = QuantumRegister(N)
        qc = QuantumCircuit(qn)
        qc.append(self._d123(), qn[:])
        super().__init__(*qc.qregs, name="D" + str(D))
        self.compose(qc, qubits=self.qubits, inplace=True)

    def _d123(self):
        """
        P0, P1, P, K1 are paramters coefficient

        qc1 (QuantumCircuit): the circuit of first layer
        qc2 (QuantumCircuit): the circuit of second layer(an alternated layer)
        qc3 (QuantumCircuit): the circuit of third layer(the remained connection)

        return: qc = qc1+qc2+qc3
        """

        qn1 = QuantumRegister(self.N)
        qc1 = QuantumCircuit(qn1)  # for the first term
        qn2 = QuantumRegister(self.N)
        qc2 = QuantumCircuit(qn2)

        qc1.name = "L1"  # for the first layer
        qc2.name = "L2"  # for the second layer

        P0 = int(self.N / self.D / 2)
        P1 = P0 * self.D
        P = self.N - P1 * 2
        K0 = int(P / self.D)
        K1 = int(P % self.D)
        K = P1 + K0 * K1

        for t in range(1, P0 + 1):
            for i in range(1, self.D + 1):
                m = (i + 2 * self.D * (t - 1) - 1) % self.N
                n = (i + 2 * self.D * t - self.D - 1) % self.N
                # print([m, n], self.Qke[m][n], "L1")
                qc1.append(
                    exp2Z(
                        self.g * self.Qke[m][n],
                    ),
                    [m, n],
                )
                if self.D != self.N / 2:
                    k = (i + 2 * self.D * t - 1) % self.N
                    # print([n, k], self.Qke[n][k], "L2")
                    qc2.append(
                        exp2Z(
                            self.g * self.Qke[n][k],
                        ),
                        [n, k],
                    )

        if P > self.D:
            for t in range(1, K1 + 1):
                m = (2 * P1 + t - 1) % self.N
                n = (2 * P1 + t + self.D - 1) % self.N
                # print([m, n], self.Qke[m][n], "L1")
                qc1.append(
                    exp2Z(
                        self.g * self.Qke[m][n],
                    ),
                    [m, n],
                )
                k = (2 * P1 + t + 2 * self.D - 1) % self.N
                # print([n, k], self.Qke[n][k], "L2")
                qc2.append(
                    exp2Z(
                        self.g * self.Qke[n][k],
                    ),
                    [n, k],
                )

        qn = QuantumRegister(self.N)
        qc = QuantumCircuit(qn)
        qc.append(qc1, qn[:])
        qc.append(qc2, qn[:])

        if self.N - 2 * K > 0:
            qn3 = QuantumRegister(self.N)
            qc3 = QuantumCircuit(qn3)
            qc3.name = "L3"  # for the third layer

            for t in range(1, self.N - 2 * K + 1):
                m = (2 * P1 + K0 * K1 + t - 1) % self.N
                n = (2 * P1 + K0 * K1 + t + self.D - 1) % self.N
                # print([m, n], self.Qke[m][n], "L3")
                qc3.append(
                    exp2Z(
                        self.g * self.Qke[m][n],
                    ),
                    [m, n],
                )
            qc.append(qc3, qn[:])

        return qc


class sCostOp(QuantumCircuit):
    """Generate the cost circuit to the single term (without out interactions) like
    exp(-i g Zi)
    N (int): the size of the circuit
    qke (np.ndarray): the coefficient list
    """

    def __init__(self, N: int, qke: np.ndarray, g: Union[float, Parameter, str]):
        if isinstance(g, str):
            g = Parameter(g)
        qn = QuantumRegister(N)
        qc = QuantumCircuit(qn)

        for i in range(N):
            qc.rz(2 * qke[i] * g, qn[i])
        super().__init__(*qc.qregs, name="SC")
        self.compose(qc, qubits=self.qubits, inplace=True)


class bdm(QuantumCircuit):
    """Generate the cost circuit with bubble dividing method
    The intend of this algorithm is to color all connection with minima depth

    Args:
        N (int): the size of qubits
        Qke (np.ndarray): the extended interation terms
        g: Union[float, Parameter, str]: parameters for optimizing
    """

    def __init__(
        self,
        N: int,
        Qke: np.ndarray,
        g: Union[float, Parameter, str],
    ):

        if isinstance(g, str):
            g = Parameter(g)

        qn = QuantumRegister(N)
        qc = QuantumCircuit(qn)
        for D in range(1, (N // 2) + 1):
            qc.append(D123(N=N, D=D, Qke=Qke, g=g), qn[:])
        super().__init__(*qc.qregs, name="BDM")
        self.compose(qc, qubits=self.qubits, inplace=True)


class costOp(QuantumCircuit):
    """
    Generate the cost operator for a general ising model such as
        sum Qke_ij Z_iZ_j + sum qke_i Z_i

    The input Qke must be half be sysmetry if not, we would transfer it
    """

    def __init__(
        self,
        eqv: EVQAA,
        method: str,
        qke: np.ndarray,
        Qke: np.ndarray,
        g: Union[float, Parameter, str],
        obj_model: str = "gmvp",
    ) -> None:
        """
        methd (str): the method to apply cost operators 'bubble': bubble dividing method
        g (float): parameter for circuit
        coeff: the coefficient generate by the costCoe class
        obj_model: specify the model for the cost operator
        """
        if isinstance(g, str):
            g = Parameter(g)
        self.g = g
        self.eqv = eqv
        self.qke = qke
        self.Qke = Qketrans(Qke)
        qn = QuantumRegister(self.eqv.N)
        qc = QuantumCircuit(qn)
        if obj_model == "gmvp":
            qc.append(sCostOp(N=self.eqv.N, qke=self.qke, g=self.g), qn[:])
            if method == "bubble":
                qc.append(bdm(N=self.eqv.N, Qke=self.Qke, g=self.g), qn[:])

        super().__init__(*qc.qregs, name="Cost")
        self.compose(qc, qubits=self.qubits, inplace=True)


def Qketrans(Qke: np.ndarray):
    """
    Change the Qke to a symmetric but the off diagonal has been amplified

    Qke: is the array that need to be converted properly for the circuit

    """

    # Qken = 2 * np.tril(Qke, -1) + np.triu(np.tril(Qke, 0), 0) + 2 * np.triu(Qke, 1)
    Qken = 2 * Qke - np.triu(np.tril(Qke, 0), 0)

    return Qken
