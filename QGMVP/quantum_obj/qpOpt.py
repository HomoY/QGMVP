# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""For quadratic programming instance generation and optimization"""
__author__ = "HMY"
__date__ = "2023-02-14"

from QGMVP.classic.CostFun import *
from QGMVP.ansatz.Ansatz import *
from QGMVP.ansatz.CostOp import *
from QGMVP.ansatz.Mixer import *
from QGMVP.ansatz.Initial import *
from QGMVP.quantum_obj.qOpt import objFun, qOptimization, statMeas
from QGMVP.optimizer.cOpt import *

# from parameters import parameter


class qpOpt:
    """
    Generate a instance of a quadratic programming quantum algorithm and solve it (function sol)
    """

    def __init__(
        self,
        eqv: EVQAA,
        p: int,
        qCircAna: dict,
        simulator: Aer,
        initBetas: Union[float, list],
        initGammas: Union[float, list],
        numIter: int,
        numOuterIter: int,
        numMeaus: int,
        q_optimizer: str,
        noise: bool = False,
        consfilter: bool = True,
        para_verbose: bool = False,
        statMeas_verbose: bool = 0,
        obj_model="gmvp",
        **kwargs,
    ) -> None:
        """
        Args:
            eqv (EVQAA): a designed binary encoding
            p (int): the layer of circuit
            simulator (Aer): the simulator of the circuit, required by qiskit
            initBetas (Union[float, list]): initial betas for qaoa mixer [intBeta1,intBeta2,intBeta3,...]
            initGammas (Union[float, list]): initial gammas for qaoa cost operator [intGamma1,intGamma2,intGamma3,...]
            numIter (int): measurements in the classical optimization cycle
            numOuterIter (int): classical optimization cycle times
            numMeaus (int): measurements after the classical optimization cycle
        """
        self.eqv = eqv
        self.numIter = numIter
        self.numOuterIter = numOuterIter
        self.numMeaus = numMeaus
        self.initBetas = initBetas
        self.initGammas = initGammas
        self.initBeta = self.initBetas[0]
        self.initGamma = self.initGammas[0]
        self.p = p
        self.simulator = simulator
        self.noise = noise
        self.consfilter = consfilter
        self.para_verbose = para_verbose
        self.statMeas_verbose = statMeas_verbose
        self.qCircAna = qCircAna
        self.q_optimizer = q_optimizer
        self.obj_model = obj_model
        if "cParas" in kwargs:
            self.cParas = kwargs["cParas"]
        if "disec" in kwargs:
            self.disec = kwargs["disec"]
        if "postMeasMeth" in kwargs:
            self.postMeasMeth = kwargs["postMeasMeth"]
        else:
            self.postMeasMeth = "smvm"

        if "circ_measure" in kwargs:
            self.circ_measure = kwargs["circ_measure"]
        else:
            self.circ_measure = True

    # @spendtime
    def CompCirc(self, qk: np.ndarray, Qk: np.ndarray):
        """
        Generate a quantum circuit of quadratic programming Optimization with a fixed sum
        constraint, and optimize the circuit to get the optimal circuit parameters.
        Quadratic programming function is for the gmvp model
                                    min 0.5* x Qk x + qk x
                                        sum x = 1
                                        x in [0,1] binary encoded by qubits
        Args:
            eqv: encoding setup
            qk: coefficient of the single order coefficient
            Qk: coefficient of the covariance matrix

        Returns:
            circ: generated circuit
            bestParams: the best parameters for this circuit
            bestEst: best estmated value
        """
        if self.obj_model == "gmvp":
            self.costFun = costsfun(sigma=Qk, mu=qk, lamb=-1.0)._costFun
            # get instance for cost operators

        # Generate circuit
        self.iniCirc = InitStatPre(
            eqv=self.eqv, Qk=Qk, qk=qk, method=self.qCircAna["ini"]
        )
        # self.iniCirc = aes(eqv=self.eqv)
        # self.iniCirc = ws(eqv=self.eqv, Qk=Qk, qk=qk)
        coeff = costCoe(
            eqv=self.eqv,
            qk=qk,
            Qk=Qk,
            obj_model=self.obj_model,
        )
        self.costCirc = costOp(
            obj_model=self.obj_model,
            eqv=self.eqv,
            method=self.qCircAna["cost"],
            qke=coeff.qke,
            Qke=coeff.Qke / 2,
            g="g",
        )
        # add mixer
        self.mixCirc = mixerOp(eqv=self.eqv, method=self.qCircAna["mixer"], b="b")

        # Get a combination of the QAOA circuit
        circ = combCircQAOA(
            n=self.eqv.N,
            name="Quadratic",
            simulator=self.simulator,
            p=self.p,
            initCirc=self.iniCirc,
            costCirc=self.costCirc,
            mixerCirc=self.mixCirc,
            measure=self.circ_measure,
        )
        self.circ = circ

    def CompCircPos(self):
        """
        Generate a quantum circuit of quadratic programming Optimization with a fixed sum in post observing process
        constraint, and optimize the circuit to get the optimal circuit parameters.
        Quadratic programming function is
                                    min 0.5* x Qk x + qk x
                                        sum x = 1
                                        x in [0,1] binary encoded by qubits
        Args:
            eqv: encoding setup
            qk: coefficient of the single order coefficient
            Qk: coefficient of the covariance matrix

        Returns:
            circ: generated circuit
            bestParams: the best parameters for this circuit
            bestEst: best estmated value
        """

        # Get a combination of the QAOA circuit
        circ = combCircQAOA(
            n=self.eqv.N,
            name="Quadratic",
            simulator=self.simulator,
            p=self.p,
            initCirc=self.iniCirc,
            costCirc=self.costCirc,
            mixerCirc=self.mixCirc,
            measure=self.circ_measure,
        )
        self.circ = circ

    def compose(
        self, rebuildCirc: bool = True, qk: np.ndarray = None, Qk: np.ndarray = None
    ):
        """
        Generate a quantum circuit of quadratic programming Optimization with a fixed sum
        constraint, and optimize the circuit to get the optimal circuit parameters.
        Quadratic programming function is
                                    min 0.5* x Qk x + qk x
                                        sum x = 1
                                        x in [0,1] binary encoded by qubits
        Args:
            eqv: encoding setup
            qk: coefficient of the single order coefficient
            Qk: coefficient of the covariance matrix

        Returns:
            circ: generated circuit
            bestParams: the best parameters for this circuit
            bestEst: best estmated value
        """
        if rebuildCirc:
            self.CompCirc(qk=qk, Qk=Qk)

        kwargs = {
            "circ": self.circ,
            "simulator": self.simulator,
            "numIter": self.numIter,
            "costFun": self.costFun,
            "eqv": self.eqv,
            "noise": self.noise,
            "consfilter": self.consfilter,
            "para_verbose": self.para_verbose,
            "statMeas_verbose": self.statMeas_verbose,
            "q_optimizer": self.q_optimizer,
            "initParamsList": initParam(
                initGamma=self.initGamma, initBeta=self.initBeta, p=self.p
            ),
        }

        # Conditionally add cParas if it exists
        if hasattr(self, "cParas"):
            kwargs["cParas"] = self.cParas

        # Conditionally add disec if it exists
        if hasattr(self, "disec"):
            kwargs["disec"] = self.disec

        # get an instance for quadratic integer programming function
        objMeanVal = objFun(**kwargs)

        copr = cOptimization()  # Get the classical optimizer
        c = qOptimization(
            costObj=objMeanVal, cOptimizer=copr
        )  # generate quantum optimizer
        self.c = c

    def opt(self, optTar: str = "mean"):
        """
        Generate a quantum circuit of quadratic programming Optimization with a fixed sum
        constraint, and optimize the circuit to get the optimal circuit parameters.
        Quadratic programming function is
                                    min 0.5* x Qk x + qk x
                                        sum x = 1
                                        x in [0,1] binary encoded by qubits
        Args:
            optTar: "mean" for mean value, or "var" for variance

        Returns:
            circ: generated circuit
            bestParams: the best parameters for this circuit
            bestEst: best estmated value
        """
        kwargs = {
            "method": self.q_optimizer,
            "initGamma": self.initGamma,
            "initBeta": self.initBeta,
            "p": self.p,
            "numOuterIter": self.numOuterIter,
            "verbose": True,
            "tyPe": 1,
            "optTar": optTar,
        }

        # Conditionally add cParas if it exists
        if hasattr(self, "cParas"):
            kwargs["cParas"] = self.cParas

        # Conditionally add disec if it exists
        if hasattr(self, "disec"):
            kwargs["disec"] = self.disec

        bestParams, bestEst = self.c.opt(**kwargs)  # start the optimization
        print("End of quantum optimization")
        return bestParams, bestEst

    def meas(self, parameters: np.ndarray):
        """Perform a measurement and their statistic estimation

        Returns:
            _type_: _description_
        """
        rc = simulate(
            circ=self.circ,
            simulator=self.simulator,
            parameters=parameters,
            numIter=self.numMeaus,
        )
        return rc

    def sol(self, optTar: str = "mean", out: str = None) -> np.ndarray:
        """Solve a quadratic function with a qaoa ansatz, and the function is
            min 0.5* x Qk x + qk x
                sum x = 1
                x in [0,1] binary encoded by qubits

        Args:
            Qk (np.ndarray): covariance matrix
            qk (np.ndarray): single interaction terms
            xk (np.ndarray): initial parameters for
            tau (float): _description_
            model (int): _description_
            param (bool): if turn on, return parameters
            out (str): control the outputs, if out = "pm": parameters and variance

        Returns:
            x (np.ndarray): the optimial position
            bestParams (np.ndarray): first p parameters are betas from beta_1 to beta_p, and the last p parameters are gammas from gamma_1 to gamma_p
        """
        # get the best parameters from the optimisation
        bestParams, _ = self.opt(optTar=optTar)
        if self.q_optimizer == "compare":
            paramList = initParam(
                initGamma=self.initGamma, initBeta=self.initBeta, p=self.p
            )
            for i in range(len(self.cParas)):
                paramList[self.cParas[i]] = bestParams[i]
            bestParams = paramList

        # The last simulations and measurement
        rc = self.meas(parameters=bestParams)

        print(
            "the parameter is:",
            bestParams,
        )

        if self.noise:
            sx, meanVal, varVal, minDict = statMeas(
                costFun=self.costFun,
                rc=rc,
                numIter=self.numMeaus,
                eqv=self.eqv,
                memory=self.postMeasMeth,
                consfilter=False,
                noise=True,
            )
            s = list(minDict.keys())[0]
            x = sx[s]
            print("mean value for the unfiltered measurement is:", meanVal)
            print(
                "the best position and cost for the unfiltered measurement is:",
                x,
                self.costFun(x),
            )

            sx, meanVal, varVal, minDict = statMeas(
                costFun=self.costFun,
                rc=rc,
                numIter=self.numMeaus,
                eqv=self.eqv,
                memory=self.postMeasMeth,
                consfilter=True,
                noise=True,
            )
            s = list(minDict.keys())[0]
            x = sx[s]
            print("mean value for the filtered measurement is:", meanVal)
            print(
                "the best position and cost for the filtered measurement is:",
                x,
                self.costFun(x),
            )

        else:
            measuredict, optdict = statMeas(
                costFun=self.costFun,
                rc=rc,
                numIter=self.numMeaus,
                eqv=self.eqv,
                memory=self.postMeasMeth,
                noise=False,
            )

            def get_first_element(input_dict):
                first_key = next(iter(input_dict))
                first_value = input_dict[first_key]
                return first_key, first_value

            # get the x of the smallest stra
            x, y_dict = get_first_element(measuredict)
            print("mean value:", optdict["meanVal"])
            print(
                "the best position:",
                x,
            )
            print("The best cost:", y_dict["cost"])
            print("The measured shots:", y_dict["freq/prob"])
        if out == "pm":
            return bestParams, optdict["meanVal"]
        elif out == "pmxf":
            return (bestParams, optdict["meanVal"], x, y_dict["cost"])

        elif out == "measopt":
            """save optimised parameters, measurement results, and the optimized result"""
            return {
                "bestParams": bestParams,
                "measuredict": measuredict,
                "optdict": optdict,
            }

        return x


def get_the_qobj(_param):
    """Get the initialized qobj, fascade for qpOpt

    Args:
        _param (params): the instance generated by parameters

    Returns:
        _type_: _description_
    """
    kwargs = {
        "eqv": _param.Enco,
        "p": _param.p,
        "qCircAna": _param.qCircAna,
        "simulator": _param.simulator,
        "initBetas": _param.initBetas,
        "initGammas": _param.initGammas,
        "numIter": _param.numIter,
        "numOuterIter": _param.numOuterIter,
        "numMeaus": _param.numMeaus,
        "q_optimizer": _param.q_optimizer,
        "noise": bool(_param.config["qiskit"]["noise"]),
        "para_verbose": _param.para_verbose,
        "statMeas_verbose": _param.statMeas_verbose,
        "circ_measure": _param.circ_measure,
        "consfilter": _param.consfilter,
        "postMeasMeth": _param.postMeasMeth,
        "obj_model": _param.obj_model,
    }

    # Conditionally add cParas and disec if they exist
    if hasattr(_param, "cParas"):
        kwargs["cParas"] = _param.cParas

    if hasattr(_param, "disec"):
        kwargs["disec"] = _param.disec

    qpObj = qpOpt(**kwargs)
    qpObj.CompCirc(qk=_param.qk, Qk=_param.Qk)

    return qpObj
