""" "
This file is for the quadratic programming main running files
"""

__author__ = "HMY"
__date__ = "2023-04-21"


from QGMVP.optimizer.cOpt import *
from QGMVP.classic.CostFun import *
from QGMVP.ansatz.CostOp import *
from QGMVP import qpOpt, get_the_qobj
from parameters import params


# import faulthandler

# faulthandler.enable()


class MulRun(object):
    """The class for the multirun"""

    def __init__(self, out: str = "pmxvf") -> None:
        self.out = out

    def standard(self, _qpOpt: qpOpt, _beta: list, _gamma: list, optTar: str = "mean"):
        """the function is used for running multiple times with fixed problem set
        The program will take a defined qpOpt, randomly change the parameters and change the qpOpt object,
        By recombining and generating the circuit, outputs a random estimation

        Args:
            _qpOpt (qpOpt): a combined class for running
            _beta: the initial betas
            _gamma: the initial gammas
            qk (np.ndarray): return array
            Qk (np.ndarray): covariance arrays

        Returns:
            Standard outputs according to the sol out parameter
        """

        _qpOpt.initGamma = _gamma
        _qpOpt.initBeta = _beta
        _qpOpt.compose(rebuildCirc=False)

        return _qpOpt.sol(out=self.out, optTar=optTar)

    def varObj(
        self,
        _Qk,
        _qk,
        _qpOpt: qpOpt,
        _beta: list,
        _gamma: list,
        optTar: str = "mean",
    ):
        """the function is used for running multiple times for varied problem sets
        The program will take a defined qpOpt, randomly change the parameters and change the qpOpt object,
        By recombining and generating the circuit, outputs a random estimation

        Args:
            _sigma: problem sigma
            _mu: problem mu
            _qpOpt (qpOpt): a combined class for running
            _beta: the initial betas
            _gamma: the initial gammas
            qk (np.ndarray): return array
            Qk (np.ndarray): covariance arrays

        Returns:
            Standard outputs according to the sol out parameter
        """
        _qpOpt.initGamma = _gamma
        _qpOpt.initBeta = _beta

        _qpOpt.compose(rebuildCirc=True, qk=_qk, Qk=_Qk)

        return _qpOpt.sol(out=self.out, optTar=optTar)


class class_fascad(object):
    """A fascade for brute force and classical optimisation"""

    def __init__(self) -> None:
        pass

    def bf(self, _param: params, run=True, *args, **kwargs):
        """
        A fascade for brute force
        Args:
            _param: a parameter class instance
            run: if True the turn the classical optimisation, else skip the process
        """

        if run:
            # show the brute force result
            bf_res = bruteForce(
                costsfun(sigma=_param.Qk, mu=_param.qk, lamb=-1.0)._costFun,
                n=_param.n,
                a=_param.Enco.amp,
                D=_param.Enco.D[0],
                bitRan=_param.Enco.bitRan,
                verbose=_param.brute_verbose,
                memory=_param.brute_memory,
            )

            return bf_res

    def qd(self, _param: params, run=True, appr_verbose=True, *args, **kwargs):
        """
        A fascade for continous quadratic programming
        Args:
            _param: a parameter class instance
            run: if True the turn the classical optimisation, else skip the process
            appr_verbose: approximate ratio verbose, True print, False not print
        """
        if run:
            qpargs = scrip(
                n=_param.n,
                Qk=_param.Qk,
                qk=_param.qk,
            )
            [P, q, G, h, A, b] = qpargs
            optX = quadOpt(*qpargs, verbose=_param.quad_verbose)

            # check the approximation good or not

            appr = optX / _param.Enco.amp
            if appr_verbose:
                print("apprxo ratio", appr, "\n")

            return optX, appr

    def bf_qd_sd(self, _param: params, run=True, appr_verbose=True):
        """A standard ouputs of brutefoce and continous quadratic programming

        Args:
            _param (params): _description_
            run (bool, optional): _description_. Defaults to True.
            appr_verbose (bool, optional): _description_. Defaults to True.
        """
        _bf = self.bf(_param=_param, run=run)
        _qd = self.qd(_param=_param, run=run, appr_verbose=appr_verbose)

        return _bf, _qd


def standard_qd(start_angle=0, outputs="pmxf"):
    """Standard test for quadratic programming with different initial parameters fix the assets combinations

    start_angle: the index for the starting angle, in case the program is interrupted and I need to restart it
    """
    _param = params()
    class_fascad().bf_qd_sd(
        _param=_param, run=_param.bf_qd_run
    )  # for the classical optimisation

    # get the quantum object
    qpObj = get_the_qobj(_param=_param)

    # for running the program in several test, first initialize the parameters list, then access to the functions to run
    if _param.runMutiple == True:
        print("Start the optimisation for", _param.runT, "times...\n")
        for i in range(_param.runT - start_angle):
            _i = i + start_angle
            print("********** The", _i, " optimisations **********")
            # create magic function
            myfun = MulRun(out=outputs).standard
            print("The parameters are:", _param.initBetas[_i], _param.initGammas[_i])
            myfun(
                _qpOpt=qpObj,
                _beta=_param.initBetas[_i],
                _gamma=_param.initGammas[_i],
                optTar=_param.optTar,
            )

            # print("Saving Data...")
            # with open(_param.pklname, "wb") as f:  # Python 3: open(..., 'wb')
            #     pickle.dump([paramsL, measL, xL, fL], f)

        print("Stop optimisations...")


if __name__ == "__main__":
    # ********** For testing files **********

    standard_qd(start_angle=0)
