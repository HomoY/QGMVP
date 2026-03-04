"""A mixer ansatz generator for Efficient Variational quantum Approximation algorithom to solve risk parity model"""

__author__ = "HMY"
__date__ = "2023-01-24"
from QGMVP.ansatz.Ansatz import *
from qiskit.circuit import Parameter

from qiskit.circuit.library import RYGate


class c2ry(QuantumCircuit):
    """This is for the two qubits controlled and target rotation of the third

    The first qubit is the target, the second and the third are controlled qubits
    t is the angle
        else it would the problem of statevector, probabiliy
    """

    def __init__(self, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)
        qn = QuantumRegister(3)
        qc = QuantumCircuit(qn)

        qc.ry(t * 0.25, qn[0])
        qc.cx(qn[1], qn[0])
        qc.ry(-t * 0.25, qn[0])
        qc.cx(qn[2], qn[0])
        qc.ry(t * 0.25, qn[0])
        qc.cx(qn[1], qn[0])
        qc.ry(-t * 0.25, qn[0])
        qc.cx(qn[2], qn[0])

        super().__init__(*qc.qregs, name="Ry(C12-> T0)")
        self.compose(qc, qubits=self.qubits, inplace=True)


class c2rx(QuantumCircuit):
    """This is for the two qubits controlled and target rx rotation of the third

    The first qubit is the target, the second and the third are controlled qubits
    t is the angle
        else it would the problem of statevector, probabiliy
    """

    def __init__(self, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)
        qn = QuantumRegister(3)
        qc = QuantumCircuit(qn)

        # qc.cry(t / 2, 1, 0)
        # qc.cx(qn[2], qn[0])
        # qc.cry(-t / 2, 1, 0)
        # qc.cx(qn[2], qn[0])

        # ccry = RYGate(t).control(2, label=None)
        # qc.append(ccry, [2, 1, 0])

        qc.rx(t * 0.25, qn[0])
        qc.cx(qn[1], qn[0])
        qc.rx(-t * 0.25, qn[0])
        qc.cx(qn[2], qn[0])
        qc.rx(t * 0.25, qn[0])
        qc.cx(qn[1], qn[0])
        qc.rx(-t * 0.25, qn[0])
        qc.cx(qn[2], qn[0])

        super().__init__(*qc.qregs, name="Rx(C12-> T0)")
        self.compose(qc, qubits=self.qubits, inplace=True)


class QE3(QuantumCircuit):
    """This is for exp(Qa+1^+QaQb-Qa^+Qb^+Qa+1 t )

    The first qubit is a+1, the second is a, and the thrid is b
    t is the angle
    """

    def __init__(self, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)

        qn = QuantumRegister(3)
        qc = QuantumCircuit(qn)

        # qc.name = "QE(0:a+1,12:ab)"
        qc.name = "QE3"
        qc.cx(qn[0], qn[1])
        qc.cx(qn[0], qn[2])

        qc.append(c2ry(t), qn[:])

        qc.cx(qn[0], qn[2])
        qc.cx(qn[0], qn[1])
        super().__init__(*qc.qregs, name="QE3")
        self.compose(qc, qubits=self.qubits, inplace=True)


class QE2(QuantumCircuit):
    """This is for qubit exciation of exp( Qa^+Qb - Qb^+Qa t)
    The first is A, and the second is B
    t is the angle
    """

    def __init__(self, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)

        qn = QuantumRegister(2)
        qc = QuantumCircuit(qn)

        qc.cx(qn[0], qn[1])
        qc.cry(
            t, control_qubit=qn[1], target_qubit=qn[0]
        )  # use the cry in the original
        qc.cx(qn[0], qn[1])
        super().__init__(*qc.qregs, name="QE2")
        self.compose(qc, qubits=self.qubits, inplace=True)


class Qe2Bin(QuantumCircuit):
    """This is used for two qubits excitation S(beta) of binary encoding"""

    def __init__(self, qblen: int, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)

        qn = QuantumRegister(
            2 * qblen
        )  # the first qblen is a1 and the last qblen is a2
        qc = QuantumCircuit(qn)

        for i in range(qblen):  # qe2
            qc.append(
                QE2(t),
                [
                    i,
                    i + qblen,
                ],
            )

        super().__init__(*qc.qregs, name="Qe2Bin")
        self.compose(qc, qubits=self.qubits, inplace=True)


class Qe23Bin(QuantumCircuit):
    """This is used for the 2 qubits and 3 qubits qubit excitation of binary encoding

    the circuit layer would be
            S^o(beta)P^o(beta)S^o(beta)S^e(beta)P^e(beta)S^e(beta)
        = S^o(beta)P^o(beta)S(beta)P^e(beta)S^e(beta)
    """

    def __init__(self, qblen: int, t: Union[float, Parameter, str]):
        """
        Args:
            qblen (int): the qubit block length of binary design
            a1 (int): the first qubit
            a2 (int): _description_
            t (Union[float, Parameter, str]): angle of the rotation
        Returns:
            the circuit of qubit excitation for the first qblen for a1 and the last qblen for the a2
        """
        if isinstance(t, str):
            t = Parameter(t)

        qn = QuantumRegister(
            2 * qblen
        )  # the first qblen is a1 and the last qblen is a2
        qc = QuantumCircuit(qn)

        for i in range(qblen // 2):  # qe2
            qc.append(
                QE2(t),
                [
                    2 * i + 1,
                    2 * i + qblen + 1,
                ],
            )

        for i in range(qblen // 2):  # first qe3
            qc.append(
                QE3(t),
                [
                    2 * i + 1,
                    2 * i,
                    2 * i + qblen,
                ],
            )

        for i in range(qblen):  # qe2
            qc.append(
                QE2(t),
                [
                    i,
                    i + qblen,
                ],
            )

        for i in range((qblen - 1) // 2):
            qc.append(
                QE3(t),
                [
                    2 * i + 2,
                    2 * i + 1,
                    2 * i + 1 + qblen,
                ],
            )

        for i in range((qblen - 1) // 2):
            qc.append(
                QE2(t),
                [
                    2 * i + 2,
                    2 * i + 2 + qblen,
                ],
            )

        super().__init__(*qc.qregs, name="Qecit")
        self.compose(qc, qubits=self.qubits, inplace=True)


class pauli_xy(QuantumCircuit):
    """This is used for Pauli XY mixer ansatz"""

    def __init__(self, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)

        qn = QuantumRegister(2)
        qc = QuantumCircuit(qn)

        qc.cx(qn[0], qn[1])
        qc.crx(
            t, control_qubit=qn[1], target_qubit=qn[0]
        )  # use the cry in the original
        qc.cx(qn[0], qn[1])

        super().__init__(*qc.qregs, name="Pxy")
        self.compose(qc, qubits=self.qubits, inplace=True)


class pauli_xxy(QuantumCircuit):
    """This is used for Pauli XXY mixer ansatz"""

    def __init__(self, t: Union[float, Parameter, str]):
        if isinstance(t, str):
            t = Parameter(t)
        qn = QuantumRegister(3)
        qc = QuantumCircuit(qn)

        # qc.name = "QE(0:a+1,12:ab)"
        qc.name = "Pxxy"
        qc.cx(qn[2], qn[0])
        qc.cx(qn[2], qn[1])

        qc.append(c2rx(t), qn[::-1])

        qc.cx(qn[2], qn[1])
        qc.cx(qn[2], qn[0])

        super().__init__(*qc.qregs, name="Pxxy")
        self.compose(qc, qubits=self.qubits, inplace=True)


class QBin(QuantumCircuit):
    """This is used for the 2 qubits and 3 qubits qubit excitation of binary encoding
    or Paulis-XY mixer ansatz,

    the circuit layer would be
            S^o(beta)P^o(beta)S^o(beta)S^e(beta)P^e(beta)S^e(beta)
        = S^o(beta)P^o(beta)S(beta)P^e(beta)S^e(beta)
    """

    def __init__(self, qblen: int, t: Union[float, Parameter, str], method: str = "QE"):
        """
        Args:
            qblen (int): the qubit block length of binary design
            a1 (int): the first qubit
            a2 (int): _description_
            t (Union[float, Parameter, str]): angle of the rotation
        Returns:
            the circuit of qubit excitation for the first qblen for a1 and the last qblen for the a2
        """
        if isinstance(t, str):
            t = Parameter(t)
        if method == "Pauli":
            q2_exci = pauli_xy
            q3_exci = pauli_xxy
        elif method == "QE":
            q2_exci = QE2
            q3_exci = QE3
        else:
            raise ValueError(f"Method {method} is not supported. Use 'QE' or 'Pauli'.")

        qn = QuantumRegister(
            2 * qblen
        )  # the first qblen is a1 and the last qblen is a2
        qc = QuantumCircuit(qn)

        for i in range(qblen // 2):  # qe2
            qc.append(
                q2_exci(t),
                [
                    2 * i + 1,
                    2 * i + qblen + 1,
                ],
            )

        for i in range(qblen // 2):  # first qe3
            qc.append(
                q3_exci(t),
                [
                    2 * i + 1,
                    2 * i,
                    2 * i + qblen,
                ],
            )

        for i in range(qblen):  # qe2
            qc.append(
                q2_exci(t),
                [
                    i,
                    i + qblen,
                ],
            )

        for i in range((qblen - 1) // 2):
            qc.append(
                q3_exci(t),
                [
                    2 * i + 2,
                    2 * i + 1,
                    2 * i + 1 + qblen,
                ],
            )

        for i in range((qblen - 1) // 2):
            qc.append(
                q2_exci(t),
                [
                    2 * i + 2,
                    2 * i + 2 + qblen,
                ],
            )

        super().__init__(*qc.qregs, name="Qecit")
        self.compose(qc, qubits=self.qubits, inplace=True)


class M123(QuantumCircuit):
    """Create the M(1), M(2), M(3) for each cycle of mixer, which is simmilar to bubble cost operator
    A similar method has been presented in the cost operator design
    """

    def __init__(
        self,
        n: int,
        N: int,
        qblen: int,
        d: int,
        b: Union[float, Parameter, str],
        method: str = "QE",
    ):
        """
        Args:
            n (int): _description_
            N (int): _description_
            allo (list): _description_
            d (int): the layer we are looked into, start from 1 to n, n is the assets number
            t (Union[float, Parameter, str]): rotation angles
            method (str): the method to use, QE or Pauli
        """
        if isinstance(b, str):
            b = Parameter(b)

        qn = QuantumRegister(N)
        # three circuit, qc1 is the first layer, and qc2 is second layer, if condition satisfied we add the third layer
        qc = QuantumCircuit(qn)
        qc1 = QuantumCircuit(qn)
        qc1.name = "M1"
        qc2 = QuantumCircuit(qn)
        qc2.name = "M2"
        # connection would depends on the depth
        p0 = int(n / d / 2)  # loop in t
        p1 = p0 * d
        p = n - p1 * 2
        k0 = int(p / d)
        k1 = int(p % d)
        k = p1 + k0 * k1

        for t in range(1, p0 + 1):
            for i in range(1, d + 1):
                # connect qubit blocks between i+2d(t-1) and i+ 2dt-d
                # m and n are assets numbers
                am = (i + 2 * d * (t - 1) - 1) % n
                an = (i + 2 * d * t - d - 1) % n
                # the position qubits of the asset m and n would be
                amq = [qblen * am, qblen * (am + 1)]
                anq = [qblen * an, qblen * (an + 1)]

                qc1.append(
                    QBin(qblen=qblen, t=b, method=method),
                    qn[amq[0] : amq[1]] + qn[anq[0] : anq[1]],
                )
                # # print([m, n], self.Qke[m][n], "L1")
                if d != n / 2:
                    ak = (i + 2 * d * t - 1) % n
                    akq = [qblen * ak, qblen * (ak + 1)]
                    # print([n, k], self.Qke[n][k], "L2")
                    qc2.append(
                        QBin(qblen=qblen, t=b, method=method),
                        qn[anq[0] : anq[1]] + qn[akq[0] : akq[1]],
                    )

        if p > d:
            for t in range(1, k1 + 1):
                am = (2 * p1 + t - 1) % n
                an = (2 * p1 + t + d - 1) % n
                # print([m, n], self.Qke[m][n], "L1")
                amq = [qblen * am, qblen * (am + 1)]
                anq = [qblen * an, qblen * (an + 1)]

                qc1.append(
                    QBin(qblen=qblen, t=b, method=method),
                    qn[amq[0] : amq[1]] + qn[anq[0] : anq[1]],
                )

                k = (2 * p1 + t + 2 * d - 1) % n
                # print([n, k], self.Qke[n][k], "L2")
                qc2.append(
                    QBin(qblen=qblen, t=b, method=method),
                    qn[anq[0] : anq[1]] + qn[akq[0] : akq[1]],
                )

        qc.append(qc1, qn[:])
        if d != n / 2:
            qc.append(qc2, qn[:])

        # if possible the last layer
        if n - 2 * k > 0:
            qc3 = QuantumCircuit(qn)
            qc3.name = "M3"  # for the third layer

            for t in range(1, n - 2 * k + 1):
                am = (2 * p1 + k0 * k1 + t - 1) % n
                an = (2 * p1 + k0 * k1 + t + d - 1) % n
                # print([m, n], self.Qke[m][n], "L3")
                amq = [qblen * am, qblen * (am + 1)]
                anq = [qblen * an, qblen * (an + 1)]
                qc3.append(
                    QBin(qblen=qblen, t=b, method=method),
                    qn[amq[0] : amq[1]] + qn[anq[0] : anq[1]],
                )
            qc.append(qc3, qn[:])

        super().__init__(*qc.qregs, name="M123")
        self.compose(qc, qubits=self.qubits, inplace=True)


class bubble(QuantumCircuit):
    """It use bubbld dividing method to create a linear depth circuit to create the full qubit excitation mixer"""

    def __init__(
        self,
        n: int,
        N: int,
        allo: list,
        b: Union[float, Parameter, str],
        distance: int = -1,
        method="QE",
    ):
        """
        Args:
            n (int): _description_
            N (int): _description_
            allo (list): _description_
            t (Union[float, Parameter, str]): rotation angles
        """
        if isinstance(b, str):
            b = Parameter(b)

        qn = QuantumRegister(N)
        qc = QuantumCircuit(qn)
        if distance == -1:
            distance = (n // 2) + 1
        # depth = 2
        print("The bubble mixer collection depth:", distance)
        if distance == (n // 2) + 1:
            print("fully connected mixer")
        elif distance == 2:
            print("nearest neigbour mixer")

        # depth = 2
        # the connection would depends on the depth
        for d in range(1, distance):
            qc.append(M123(n=n, N=N, qblen=allo[0], d=d, b=b, method=method), qn[:])

        super().__init__(*qc.qregs, name="bubble")
        self.compose(qc, qubits=self.qubits, inplace=True)


class mixerOp(QuantumCircuit):
    def __init__(
        self, eqv: EVQAA, method: str, b: Union[float, Parameter, str], **kwargs
    ) -> None:
        """this class, we will defined a set of method to design a mixer, like Trotterization, bubble pairing, etc

        Args:
            eqv (EVQAA): _description_
            method (str): Trotter is to use Trotter decomposition
            b (Union[float, Parameter, str]): the controlled angle, a parameter
            distance (int): when choose the method "Bubble", this parameter need to be included
            ls (int, optional): when choose the method "Trotter" how many layers you want to divide,the default is half of asset size, which is defined in the trotter class. Defaults to False.
        """
        if isinstance(b, str):
            b = Parameter(b)

        qn = QuantumRegister(eqv.N)
        qc = QuantumCircuit(qn)

        if method["name"] == "Bubble":
            try:
                self.distance = method["distance"]
                qc.append(
                    bubble(
                        n=eqv.n,
                        N=eqv.N,
                        allo=eqv.allo,
                        b=b,
                        distance=self.distance,
                        method="QE",
                    ),
                    qn[:],
                )
            except:
                raise Exception("missing parameters in method")
        elif method["name"] == "BubblePauli":
            try:
                self.distance = method["distance"]
                qc.append(
                    bubble(
                        n=eqv.n,
                        N=eqv.N,
                        allo=eqv.allo,
                        b=b,
                        distance=self.distance,
                        method="Pauli",
                    ),
                    qn[:],
                )
            except:
                raise Exception("missing parameters in method")

        super().__init__(*qc.qregs, name="Mixer")

        self.compose(qc, qubits=self.qubits, inplace=True)
