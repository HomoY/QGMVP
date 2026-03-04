"""A class for initial state preparation"""

__author__ = "HMY"
__date__ = "2023-01-24"

from typing import List
from QGMVP.ansatz.Ansatz import *
from QGMVP import ClassRandSam
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, RYGate, PhaseGate, XGate, MCXGate
from qiskit.circuit import Gate
from QGMVP.optimizer.cOpt import scrip, quadOptInitial
import random


class PhaseRYGate(Gate):
    def __init__(self, phase, theta):
        super().__init__("pha_ry", 1, [phase, theta])

    def _define(self):
        qc = QuantumCircuit(1)
        qc.append(PhaseGate(self.params[0]), [0])
        qc.append(RYGate(self.params[1]), [0])
        self.definition = qc


class PhaseRYGateMulCont(Gate):
    def __init__(self, phase, theta, contstr):
        """_summary_

        Args:
            phase (_type_): The phase of the phase gate
            theta (_type_): The angle of the controlled RY gate
            conts (_type_): The control string
        """
        self.contstr = contstr
        super().__init__(
            "pha_ry_mc", len(contstr) + 1, [phase, theta], label="pha_ry_mc:" + contstr
        )

    def _define(self):
        qcl = len(self.contstr) + 1
        qc = QuantumCircuit(qcl)

        for i in range(len(self.contstr)):
            if self.contstr[i] == "0":
                qc.x([i])

        qc.append(
            MCMT(
                PhaseRYGate(phase=self.params[0], theta=self.params[1]),
                num_ctrl_qubits=len(self.contstr),
                num_target_qubits=1,
            ),
            list(range(qcl)),
        )

        for i in range(len(self.contstr)):
            if self.contstr[i] == "0":
                qc.x([i])

        self.definition = qc


class MulCont(Gate):
    def __init__(self, control_qubits, target_qubits, contstr):
        """Multi control and multi taget gate

        Args:
            control_qubits (_type_): The control qubits
            target_qubits (_type_): The target qubits
            conts (_type_): The control string
        """
        self.control_qubits = control_qubits
        self.target_qubits = target_qubits
        self.contstr = contstr

        if control_qubits != len(contstr):
            raise ValueError(
                "The length of the control qubits and the control string should be the same!"
            )

        super().__init__("mul_cont", control_qubits + target_qubits, [])

    def _define(self):
        num_qubits = self.control_qubits + self.target_qubits
        qc = QuantumCircuit(num_qubits)

        # Apply X gates to control qubits if needed
        for i in range(len(self.contstr)):
            if self.contstr[i] == "0":
                qc.x([i])

        # Create and append the MCMT gate
        mcmt_gate = MCMT(
            XGate(),
            num_ctrl_qubits=self.control_qubits,
            num_target_qubits=self.target_qubits,
        )
        qc.append(mcmt_gate, list(range(num_qubits)))

        # Apply X gates to control qubits if needed
        for i in range(len(self.contstr)):
            if self.contstr[i] == "0":
                qc.x([i])

        self.definition = qc


class MulTarg(Gate):
    def __init__(self, control_qubits, target_qubits):
        """Multi control and multi taget gate

        Args:
            control_qubits (_type_): The control qubits
            target_qubits (_type_): The target qubits
            conts (_type_): The control string
        """
        self.control_qubits = control_qubits
        self.target_qubits = target_qubits

        super().__init__("mul_targ", control_qubits + target_qubits, [])

    def _define(self):
        num_qubits = self.control_qubits + self.target_qubits
        qc = QuantumCircuit(num_qubits)

        # Create a multi-controlled X gate
        mcx_gate = MCXGate(num_ctrl_qubits=self.control_qubits)

        # Apply the multi-controlled target X gate to the circuit
        for target in range(
            self.control_qubits, self.target_qubits + self.control_qubits
        ):
            qc.append(mcx_gate, list(range(self.control_qubits)) + [target])

        self.definition = qc


class InitStatPre(QuantumCircuit):
    """
    Construct an initial circuit

    """

    def __init__(self, eqv: EVQAA, method: str = "maxbias", **kwargs) -> None:
        # super().__init__(n=n, enco=enco) # get the content from
        """
        method (str): method to get your initial state circuit,
            'aes': approximate equal distribution for the binary encoding,
            'ws': warm starting approximation
            'rd': random generated with constraint
            'maxbias': all in the first qubits blocks
        """

        self.pos = InitStatPreMed(eqv=eqv, method=method, **kwargs).pos
        qc = self.singlestatepre(eqv, self.pos)

        if "rd" in method:
            super().__init__(*qc.qregs, name="rd")
        else:
            super().__init__(*qc.qregs, name=method)

        self.compose(qc, qubits=self.qubits, inplace=True)

    def singlestatepre(self, eqv: EVQAA, pos: List):
        """If you want to prepare for a single state
        (worked for the 'aes', "rd', 'ws', 'maxbias','wsrrd') methods

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        qn = QuantumRegister(eqv.N)
        qc = QuantumCircuit(qn)
        # Step 3: change to string my order is for block 1[1st binary qubits, 2nd binary, ....], block 2, block 3
        s = [bin(pos[i])[2:][::-1] for i in range(len(pos))]

        # Step 4: place with the X gate
        for i in range(len(s)):
            [qc.x(qn[eqv.allo[0] * i + j]) for j in range(len(s[i])) if s[i][j] == "1"]
        return qc


class InitStatPreMed(object):
    """
    Get pos of initial state according to different methods

    """

    def __init__(self, eqv: EVQAA, method: str = "maxbias", **kwargs) -> None:
        """
        method (str): method to get your initial state circuit,
            'aes': approximate equal distribution for the binary encoding,
            'ws': warm starting approximation
            'wsrrd': warm starting approximation with random rounding
            'rd': random generated with constraint
            'maxbias': all in the first qubits blocks
        """
        self.eqv = eqv

        if method == "aes":
            self.pos = self.aes()
        elif method == "maxbias":
            self.pos = self.maxbias()
        elif method == "ws" and "Qk" in kwargs and "qk" in kwargs:
            self.pos = self.ws(Qk=kwargs["Qk"], qk=kwargs["qk"])
        elif method == "wsrrd" and "Qk" in kwargs and "qk" in kwargs:
            self.pos = self.wsrrd(Qk=kwargs["Qk"], qk=kwargs["qk"])
        elif "rd" in method:
            if "u" in method["rd"]:
                u = method["rd"]["u"]
            else:
                u = -1
            if "l" in method["rd"]:
                l = method["rd"]["l"]
            else:
                l = 0
            if "constrined" in method["rd"]:
                constrined = method["rd"]["constrined"]
            else:
                constrined = False
            self.pos = self.rd(u=u, l=l, constrined=constrined)

    def maxbias(self, asset=0):
        """generate initial state only the first asset qubit blocks is full, while the rest would be empty
        eqv (EVQAA): the information about encoding
        """
        pos = [0 for _ in range(self.eqv.n)]
        pos[asset] = self.eqv.D[0]

        return pos

    def aes(self):
        # Step 1: get the order
        p = int(
            1 / self.eqv.amp[0] - (1 // (self.eqv.amp[0] * self.eqv.n)) * self.eqv.n
        )  # if we only care about the first
        s1 = int(1 // (self.eqv.amp[0] * self.eqv.n)) + 1
        s2 = s1 - 1
        pos = [s1 for _ in range(p)] + [s2 for _ in range(p, self.eqv.n)]

        return pos

    def ws(
        self,
        Qk: np.ndarray = None,
        qk: np.ndarray = None,
    ):
        # Step 1: get the continous quadratic solution
        qpargs = scrip(
            n=self.eqv.n,
            Qk=Qk,
            qk=qk,
        )
        optX = quadOptInitial(*qpargs)

        # Step 2: rank the decimal part, increase the remaining part
        pos = optX // self.eqv.amp
        pos = np.array([is_integer(pos[i]) for i in range(len(pos))])

        remainInte = self.eqv.D[0] - sum(pos)
        remainDeci = optX / self.eqv.amp - pos
        sortPos = np.argsort(-remainDeci)
        for i in range(remainInte):
            pos[sortPos[i]] += 1

        return pos

    def wsrrd(
        self,
        Qk: np.ndarray = None,
        qk: np.ndarray = None,
    ):
        # Step 1: get the continous quadratic solution
        qpargs = scrip(
            n=self.eqv.n,
            Qk=Qk,
            qk=qk,
        )
        optX = quadOptInitial(*qpargs)

        # Step 2: randomly rounding
        pos = optX // self.eqv.amp
        pos = np.array([is_integer(pos[i]) for i in range(len(pos))])

        remainInte = self.eqv.D[0] - sum(pos)
        rrdPos = random.sample(range(self.eqv.n), remainInte)
        for i in range(remainInte):
            pos[rrdPos[i]] += 1

        return pos

    def rd(self, l: int = 0, u: int = -1, constrined=False):
        if u == -1:
            u = self.eqv.D[0]
        # Step 1: get the continous quadratic solution
        if constrined:
            randpos = ClassRandSam(
                n=self.eqv.n, l=l, u=u, total=self.eqv.D[0]
            ).get_constrained_sum_sample_pos()
        elif not constrined:
            randpos = ClassRandSam(n=self.eqv.n, l=l, u=u).get_bounded_sample_pos()

        return randpos


def is_integer(number, tolerance=1e-9):
    if abs(number - round(number)) >= tolerance:
        raise ValueError(f"The number {number} is not an integer.")
    return round(number)
