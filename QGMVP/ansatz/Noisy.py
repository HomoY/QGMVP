# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""For generating noise model into the circuit."""
__author__ = "HMY"
__date__ = "2023-05-03"

from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)
import numpy as np
from QGMVP.ansatz.Ansatz import EVQAA


class noiseModel(object):
    """generate different error"""

    def __init__(self, qubit: int):
        self.qubit = qubit

    def readout(self):
        # ibm_vigo fake end, and it contains single-qubit erros, two qubit gate errors and single-qubit readout errors
        noise_model = NoiseModel()
        # Single qubit errors
        # deplorising channel
        p_dep = 0.05
        error = pauli_error([("X", p_dep), ("I", 1 - p_dep)])
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
        # Create readout errors
        # (0,): [array([0.9234, 0.0766]), array([0.0736, 0.9264])]
        # (1,): [array([0.9904, 0.0096]), array([0.0354, 0.9646])]
        # (2,): [array([0.994, 0.006]), array([0.0232, 0.9768])]
        # (3,): [array([0.984, 0.016]), array([0.027, 0.973])]
        # (4,): [array([0.9774, 0.0226]), array([0.044, 0.956])]
        # for simplicity, I just use the average
        p0given1 = 0.026
        p1given0 = 0.041

        ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])

        return noise_model

    def thermal(self):
        # Thermal relaxiation
        # ref: https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html
        # T1 and T2 values for all qubits
        T1s = np.random.normal(
            50e3, 10e3, self.qubit
        )  # Sampled from normal distribution mean 50 microsec
        T2s = np.random.normal(
            70e3, 10e3, self.qubit
        )  # Sampled from normal distribution mean 70 microsec

        # Truncate random T2s <= T1s
        T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(self.qubit)])

        # Instruction times (in nanoseconds)
        time_u1 = 0  # virtual gate
        time_u2 = 50  # (single X90 pulse)
        time_u3 = 100  # (two X90 pulses)
        time_cx = 300
        time_reset = 1000  # 1 microsecond
        time_measure = 1000  # 1 microsecond

        # QuantumError objects
        errors_reset = [
            thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)
        ]
        errors_measure = [
            thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)
        ]
        errors_u1 = [
            thermal_relaxation_error(t1, t2, time_u1) for t1, t2 in zip(T1s, T2s)
        ]
        errors_u2 = [
            thermal_relaxation_error(t1, t2, time_u2) for t1, t2 in zip(T1s, T2s)
        ]
        errors_u3 = [
            thermal_relaxation_error(t1, t2, time_u3) for t1, t2 in zip(T1s, T2s)
        ]
        errors_cx = [
            [
                thermal_relaxation_error(t1a, t2a, time_cx).expand(
                    thermal_relaxation_error(t1b, t2b, time_cx)
                )
                for t1a, t2a in zip(T1s, T2s)
            ]
            for t1b, t2b in zip(T1s, T2s)
        ]

        # Add errors to noise model
        noise_thermal = NoiseModel()
        for j in range(self.qubit):
            noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
            noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
            noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
            noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
            noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(self.qubit):
                noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

        return noise_thermal

    def bitflip(self):
        # bit flip noise
        # Example error probabilities
        p_reset = 0.0001
        p_meas = 0.0001
        p_gate1 = 0.0001

        # QuantumError objects
        error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
        error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
        error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
        error_gate2 = error_gate1.tensor(error_gate1)

        # Add errors to noise model
        noise_bit_flip = NoiseModel()
        noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
        noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
        noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
        noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
        # # gate noise

        # simulator = AerSimulator(noise_model=noise_bit_flip)
        # ref:https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html

        return noise_bit_flip
