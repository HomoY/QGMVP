"""
This is a file for all input parameters
 n (int): asset number
 N (int): qubit number
 rc (bool): random configuration, if True, generate a random parameter set; False, find the corresponding parameters from expParam file; if not find will prompt error

 p (int): the layer of the circuit, each layer includes a cost operator and a mix operator
 initGamma (np.ndarray): 1*p1, p1<p the initial parameter gammas in cost operator
 initBeta (np.ndarray): 1*p1, p1<p, the initial parameter betas in mix opertor

 rangeGamma (np.ndarray): np.nparray([float, float]) the range change of gamma paramter
 rangeBeta (np.ndarray): np.nparray([float, float]) the range change of beta paramter

 numIter (int): the number of iterations in each circuit measurement
 numOuterIter (int): the number of iterations classical optimization loops
 numMeaus (int): the number of measurements in the last circuit, which is usually higher than numIter
"""

import numpy as np
from qiskit_aer import AerError, Aer, AerSimulator

# from qiskit.providers.fake_provider import FakeVigo
from QGMVP import readMulRun, initGene, evqaa, dataDfDict, modelGene
from QGMVP.ansatz.Noisy import noiseModel
import QGMVP.ansatz as az

import json
from typing import Union
import random
import os


# param_verbose = True
def SimulSet(Qjson: dict):
    """This is used for generating the qiskit simulation parameters

    Args:
        Qjson (dict): qiskit parameters

    Returns:
        simulator: simulator with pre-setup
    """

    if "noise" in Qjson:
        if Qjson["noise"]:
            if Qjson["noise"]["model"] == "thermal":
                try:
                    noise_model = noiseModel(
                        qubit=int(Qjson["noise"]["qubit"])
                    ).thermal()
                    simulator = AerSimulator(noise_model=noise_model)
                    # ref:https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html
                except AerError as e:
                    print(e)
        else:
            simulator = Aer.get_backend("aer_simulator_statevector")
            simulator.set_options(
                seed_simulator=Qjson["seed_simulator"],
            )

    if "GPU" in Qjson:
        try:
            simulator.set_options(
                method=Qjson["GPU"]["method"],
                device="GPU",
                seed_simulator=Qjson["seed_simulator"],
            )
            # simulator.set_options(device="GPU")
        except AerError as e:
            print(e)
    if "thread" in Qjson:
        simulator.set_options(max_parallel_threads=Qjson["thread"])

    return simulator


class params:
    def __init__(self, sid=None, verbose: bool = True, save=False) -> None:
        if verbose:
            print("Start reading parameters...")
        """
        Get the session id and read parameters from the json files
        """
        if sid == None:
            self.sid = "test.json"
        else:
            self.sid = sid

        f = open(self.sid)
        data = json.load(f)
        data = data["parameters"]
        """
        Get the save folders
        """
        self.save = data["save"]

        """
        Set the random seeds for simulation
        """

        seed = data["seed"]
        random.seed(seed)  # for the initial state preparation
        np.random.seed(seed)  # for the parameter generator

        """
        Run controls
        """
        if "bf_qd_run" in data:
            self.bf_qd_run = data["bf_qd_run"]
        if "run" in data:
            self.runMutiple = data["run"]["runMutiple"]  # If run multiple times
            self.runT = data["run"]["runT"]  # running times

        """
        System information
        Enco: Encoding parameters
        For more about Encoding information, please see all explanations in expParam/__init__.py
        """
        if "enco" in data:
            self.n = data["enco"]["n"]
            self.N = data["enco"]["N"]
            self.Enco = evqaa(n=self.n, N=self.N)

            """
            The below are for the GMVP model's covariance matrix with mu all zero
            """
            if "object" in data:
                self.get_object(**data["object"])

        """
        Ansart Parameters
        
        qCircAna (dict): "ini": initial circuit, "maxbias", "ws", "aes", "rd"
                        for "rd": you can also define upper and lower as parameters u and l
                         "cost": cost operator, "bubble": bubble design
                         "mixer": "name": name of cost operator, including "Bubble"(need distance parameter, -1 means the M_fe, 1 means M_nn)
        optTar (str): what to optimise, if mean, your objective function is mean, var is variance, lang is lang
        
        You can also secify what kind of mixing and cost operator to optimize
        classical optimzers are Ipopt, SPSA, COBYLA, Powell, Annealing, Adam, L-BFGS-B, compare(layerwise optimisation)
                                var is a special one just for checking the value 
                                gird: is a scanning of the surface
        cParas (list): [beta0, beta1,...,gamma0,...]
                                
        Mutiple Simulation options also provided(only for the main.py)
        """
        self.qCircAna = data["circuit"]["qCircAna"]

        """
        Circuit parameters
        p (int): layer of the QAOA
        initGamma, initBeta: initial gamma, and beta the range of both is [0,2pi]
        """
        if "layer" in data["circuit"]:
            if "p" in data["circuit"]["layer"]:
                self.p = data["circuit"]["layer"]["p"]
            if "CirParams" in data["circuit"]["layer"]:
                self.initGammas, self.initBetas = self.get_params(
                    run=self.runT, **data["circuit"]["layer"]["CirParams"]
                )

        # The circuit measurements, default is True
        if "circ_measure" in data["circuit"]:
            self.circ_measure = data["circuit"]["circ_measure"]
        else:
            self.circ_measure = True

        """
        Optimisation  parameters (must fixed):

        numIter (int): the measurement times for each optimisation iterartions
        numOuterIter (int): the maximum number of iterations for the program, only relate to the annealing, should wrap it
        numMeaus (int): the measurement times for estimating the mean after optimisation, it might be larger than the numIter for an accurate description
        """

        self.q_optimizer = data["optimizer"]["method"]
        self.optTar = data["optimizer"]["optTar"]

        self.numIter = data["optimizer"]["numIter"]
        self.numOuterIter = data["optimizer"][
            "numOuterIter"
        ]  # only relate to the annealing, should wrap it

        # parameters for gird and compare
        if self.q_optimizer == "gird" or self.q_optimizer == "compare":
            if "cParas" in data["optimizer"] and "disec" in data["optimizer"]:
                self.cParas = data["optimizer"]["cParas"]
                self.disec = data["optimizer"]["disec"]
        # self.cParas = [4]
        # self.disec = 1000

        # Post processing parameters
        self.numMeaus = data["postprocess"]["numMeaus"]
        self.postMeasMeth = data["postprocess"]["postMeasMeth"]

        if "consfilter" in data:
            self.consfilter = data["consfilter"]
        else:
            self.consfilter = False

        """
        Simulator Setup, get from the config json file
        """

        with open(
            f"{os.getcwd()}/config/" + data["simulator"]["conpath"], "r"
        ) as f:  # if noise change this to the conf_noise.josn
            self.config = json.load(f)

        self.simulator = SimulSet(Qjson=self.config[data["simulator"]["set"]])

        """
        Verbose parameters, True, display; False, not display 

        
        en_para_verbose: encoding parameters 
        circ_para_verbose: circ parameters, 
        simu_para_verbose: simulator and options parameters
        qc_para_verbose: qudratic programming circuit running parameters 
        brute_verbose: show brute force results 
        opt_verbose: show the optimized parameters
        para_verbose: If True print the every optimised parameters, the printed parameters will be [beta_0, beta_1, ...] and then [gamma_0, gamma_1, ...]
        statMeas_verbose: show the optimized result: 0 means no verbose, 1 means only mean value, 2 includes mean, min max, 3 adds corresponding configuration and other related information
        paraExpQuaDump:bool, True will dump all variables to a file named with filename_befr_q.pkl in the filepath
        """

        self.en_para_verbose = True
        self.circ_para_verbose = True
        self.simu_para_verbose = True
        self.qc_para_verbose = True

        self.brute_verbose = True
        self.brute_memory = True
        self.quad_verbose = True

        self.quant_verbose = True
        self.classi_verbose = True

        self.para_verbose = False
        self.statMeas_verbose = 1

        self.paraExpQuaDump = True
        self.noise_verbose = False

        if verbose:
            self.VerbParams()

    def update_enco(self, n: int, N: int):
        """Update the enco

        Args:
            rand (bool, optional): If get random sigma, mu, will influence Qk, and qk. Defaults to False.
            verbose (bool, optional): If print out the objective problems. Defaults to True.
        """

        self.n = n
        self.N = N
        self.Enco = evqaa(n=self.n, N=self.N)

    def get_object(self, method="rand", verbose=True, amplify=200, *args, **kwargs):
        """Get the objective problem

        Args:
            rand (bool, optional): If get random sigma, mu, will influence Qk, and qk. Defaults to False.
            verbose (bool, optional): If print out the objective problems. Defaults to True.
        """
        self.obj_model = kwargs["name"]
        self.sigmas = []
        self.mus = []

        if method == "test_sets":
            sigma, mu = modelGene(n=self.n, method=method, verbose=verbose, **kwargs)
            self.sigmas = [sigma[i] * amplify for i in range(self.runT)]
            self.mus = [mu[i] * amplify for i in range(self.runT)]
        else:
            for _ in range(self.runT):
                sigma, mu = modelGene(
                    n=self.n, method=method, verbose=verbose, **kwargs
                )
                self.sigmas.append(sigma * amplify)
                if mu == False:
                    self.mus.append(np.array([0.0 for _ in range(self.n)]))
                else:
                    self.mus.append(mu)

        self.sigma = self.sigmas[0]
        self.mu = self.mus[0]

        # Rewrite the parameters name as qk and Qk for the later use
        self.qk = self.mu
        self.Qk = self.sigma

    def update_object(self, sigma: np.ndarray, mu: np.ndarray, verbose=True):
        """Get the objective problem

        Args:
            rand (bool, optional): If get random sigma, mu, will influence Qk, and qk. Defaults to False.
            verbose (bool, optional): If print out the objective problems. Defaults to True.
        """
        params_verbose.update_object(sigma=sigma, mu=mu, verbose=verbose)

        self.sigma = sigma
        self.mu = mu

        # Rewrite the parameters name as qk and Qk for the later use
        self.qk = self.mu
        self.Qk = self.sigma

    def get_params(
        self,
        run: int = 100,
        rand: bool = False,
        file=False,
        test=False,
        fix=True,
        verbose=False,
    ):
        if rand:
            gamms = []
            betas = []
            for _ in range(run):
                tmpGamma, tmpBeta = initGene(
                    initGamma=[],
                    initBeta=[],
                    p=self.p,
                    bound_verbose=False,
                )
                gamms.append(tmpGamma)
                betas.append(tmpBeta)
        elif test == "t1":
            # 2 layers with the first layer of the optimised and rest the
            gamms = [[0.9356997541684998, 0.0], [0.9356997541684998, 0.0]]
            betas = [[1.4483044984859965, 0.0], [1.4483044984859965, 0.0]]
        elif test == "t2":
            gamms = [[1.9110455714553918, 4.555827060763239e-06, 0.0]]
            betas = [[4.826129525553255, 5.293955920339377e-23, 0.0]]
        elif file:
            # read files and if the files's parameter is shorter than the p, then add the last parameter randomly
            c = readMulRun(filename=file)
            # c = dataDf(name=["bestParams", "meanVal", "varVal", "x", "f(x)"], data=c)
            c = dataDfDict(name=["bestParams"], data=c)
            gamms = []
            betas = []
            for i in c["bestParams"][0:run].tolist():
                data_p = len(i) // 2
                append_p = self.p - data_p
                if append_p < 0:
                    raise Exception("wrong old optimised parameters")
                tmpGamma = i[data_p:].tolist() + [0.0 for _ in range(append_p)]
                tmpBeta = i[:data_p].tolist() + [0.0 for _ in range(append_p)]
                betas.append(tmpBeta)
                gamms.append(tmpGamma)
        elif fix:
            # gamms = [np.pi / 3 for _ in range(run)]
            # betas = [np.pi / 4 for _ in range(run)]
            gamms = [0.0 for _ in range(run)]
            betas = [0.0 for _ in range(run)]
        if verbose:
            print("Initial parameters:")
            print("Beta:", betas)
            print("Gamma:", gamms)

        return gamms, betas

    # def UpdLayer(
    #     self,
    #     p: int = None,
    #     initGamma: Union[float, np.ndarray] = None,
    #     initBeta: Union[float, np.ndarray] = None,
    # ):
    #     if p == True:
    #         pass

    def VerbParams(self):
        """ "
        Verbose parameters, True, display; False, not display

        en_para_verbose: encoding parameters
        circ_para_verbose: circ parameters
        simu_para_verbose: simulator and options parameters
        qc_para_verbose: qudratic programming circuit running parameters
        brute_verbose: show brute force results
        opt_verbose: show the optimized parameters
        statMeas_verbose: show the optimized result: 0 means no verbose, 1 means only mean value, 2 includes mean, min max, 3 adds corresponding configuration
        paraExpQuaDump:bool, True will dump all variables to a file named with filename_befr_q.pkl in the filepath
        """

        if self.en_para_verbose == True:
            print("Encoding parameters:")
            print("n =", self.n)
            print("N =", self.N)
            print("\n")

        if self.circ_para_verbose == True:
            print("Circ parameters:")
            print("p (the layer of circuit) = ", self.p)
            print("The initial state prepare is", self.qCircAna["ini"])
            print("The cost operator is", self.qCircAna["cost"])
            print("The mixing operator is", self.qCircAna["mixer"])

            print("\n")

        if self.simu_para_verbose == True:
            print("Simulator parameters:")
            print("simulator: ", self.simulator)
            print(self.simulator.options)
            print("\n")

        if self.qc_para_verbose == True:
            print("Qudratic programming circuit running parameters:")
            print("The classical optimzer for the circuit:", self.q_optimizer)
            if self.q_optimizer == "gird" or self.q_optimizer == "compare":
                # variance = True
                # print("Turn on variance:", variance)
                print("gird parameters cParas:", self.cParas, "disec", self.disec)
            print("numIter(each measurement of circuit) = ", self.numIter)
            print("numOuterIter(classical optimization cycle) = ", self.numOuterIter)
            print(
                "numMeaus(After fixed parameter in circuit, the total measurements) = ",
                self.numMeaus,
            )
            print("Meaure verbose is", self.statMeas_verbose)
            print("Parameter verbose is", self.para_verbose)
            print("The size of system:", 2**self.N)

            print("\n")
        if self.noise_verbose == True and self.config["qiskit"]["noise"]:
            print("The noise parameters of the system:", self.config)
            print("The noise filter is", self.consfilter)
            print("\n")


class params_verbose(object):
    """A summary of parameter verbose"""

    def __init__() -> None:
        pass

    def update_object(sigma: np.ndarray, mu: np.ndarray, verbose=True, *args, **kwargs):
        if verbose:
            print("Update objective function....")
            print("New sigma:")
            print(sigma)
            print("New mu:")
            print(mu)
