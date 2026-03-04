# How to install
- Clone the repo
- `cd` to the repo folder and run `pip install .` from the cloned repo under your virtual environment

# Examples
Run `main.py`, the parameters for quantum ansatz is included in `test.json`.

Below is an example of the `test.json` configuration file with explanations for each parameter block:

## Basic Configuration
```json
{
    "name": "GMVPOpt",
    "parameters": {
        "save": true,
        "seed": 42,
        ...
    }
}
```
- `name`: Name of the simulation
- `save`: Whether to save the simulation results
- `seed`: Random seed for reproducible results

## Encoding Parameters
```json
"enco": {
    "n": 5,
    "N": 15
}
```
- `n`: The number of assets in the portfolio
- `N`: The number of qubits (must satisfy: N/n is an integer, representing the binary block length)

## Problem Instance Configuration
```json
"object": {
    "name": "gmvp",
    "method": "test_sets",
    "test_sets": "rand_assets_short_market_n5N15_slec",
    "rand": false,
    "amplify": 1,
    "verbose": true
}
```
- `name`: Type of optimization problem ("gmvp" for Global Minimum Variance Portfolio)
- `method`: Instance generation method ("test_sets" uses pre-generated market data)
- `test_sets`: Specific test dataset to use (required when method is "test_sets")
- `rand`: Whether to use purely random generation instead of market data
- `amplify`: Amplification factor for the problem coefficients
- `verbose`: Whether to print detailed instance information

## Classical Preprocessing
```json
"bf_qd_run": true,
"run": {
    "runT": 100,
    "runMutiple": true
}
```
- `bf_qd_run`: Whether to run brute-force and quadratic programming solutions for comparison
- `runT`: Number of instances to simulate
- `runMutiple`: Whether to run multiple instances

## Quantum Circuit Configuration
```json
"circuit": {
    "qCircAna": {
        "ini": "ws",
        "cost": "Bubble",
        "mixer": {
            "name": "Bubble",
            "distance": 2
        }
    },
    "layer": {
        "p": 2,
        "CirParams": {
            "rand": true,
            "fix": false,
            "file": false,
            "verbose": true
        }
    },
    "circuit_measure": true
}
```

### Initial State and Operators
- `ini`: Initial state preparation method
  - "ws": Ranked warm-start
  - "maxbias": Max-biased state
  - "aes": Approximate equal-weighted state
  - "rd": Random-weighted state
- `cost`: Cost operator architecture (default: "Bubble")
- `mixer.name`: Mixing operator architecture (default: "Bubble" for qubit excitation， "BubblePauli" for parity mixer(XY-mixer) and XXY mixer)
- `mixer.distance`: Maximum distance between assets (L=1 for nearest-neighbor mixing)

### Circuit Layers and Parameters
- `p`: Number of QAOA layers (each layer contains one cost and one mixing operator)
- `CirParams.rand`: Whether to generate circuit parameters randomly
- `CirParams.fix`: Whether to use fixed parameters
- `CirParams.file`: Whether to read parameters from the previous simulations for the initial parameters (for layerwised optimization)
- `CirParams.verbose`: Whether to print parameter details
- `circuit_measure`: Whether to perform post-optimization measurements

## Optimization Configuration
```json
"optimizer": {
    "method": "Annealing",
    "optTar": "mean",
    "numIter": 16,
    "numOuterIter": 2000
}
```
- `method`: Optimization algorithm ("Annealing" for dual annealing, "COBYLA" for COBYLA)
- `optTar`: Optimization target ("mean" optimizes expectation values)
- `numIter`: Number of shots for estimating expectation values
- `numOuterIter`: Maximum number of expectation value evaluations allowed

## Post-processing Configuration
```json
"postprocess": {
    "numMeaus": 2000,
    "postMeasMeth": "measuredict"
},
"consfilter": false
```
- `numMeaus`: Number of measurement shots for final post-optimization evaluation
- `postMeasMeth`: Final measurement method ("measuredict" outputs shot measurement dictionary)
- `consfilter`: Whether to apply capital constraint filtering during optimization

## Simulator Configuration
```json
"simulator": {
    "conpath": "conf_cpu.json",
    "set": "qiskit"
}
```
- `conpath`: Configuration file name for simulation settings
- `set`: Parameter set within the specified configuration file

There are also configs in `./config` folder including:
`conf_cpu.json` for CPU simulations,
`conf_gpu.json` for GPU simulations,
`conf_noise.json` for noise simulations (only support CPU simulation).

# Reference

```
Yuan, H., Long, C. K., Lepage, H. V., & Barnes, C. H. W. (2024). Quantifying the advantages of applying quantum approximate algorithms to portfolio optimisation. arXiv preprint arXiv:2410.16265. https://arxiv.org/abs/2410.16265
```

```bibtex
@misc{yuan2024quantifyingadvantagesapplyingquantum,
      title={Quantifying the advantages of applying quantum approximate algorithms to portfolio optimisation}, 
      author={Haomu Yuan and Christopher K. Long and Hugo V. Lepage and Crispin H. W. Barnes},
      year={2024},
      eprint={2410.16265},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2410.16265}
}
```


