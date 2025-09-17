# SimQC-TTN: Quantum circuit simulation with DMRG using Tree Tensor Networks

**SimQC-TTN** is a toolkit for simulating quantum circuits using **Tree Tensor Networks (TTN)** as the state representation and **Density-Matrix Renormalization Group (DMRG)** for compression. It includes ready-to-run examples for **QAOA** (native and flexible/chunked) and **random circuits**, plus Jupyter notebooks for interactive use.

## Features
- **QAOA simulators**
  - **Native:** build each full layer, then compress
  - **Flexible:** interleave batches of two-qubit gates with compression (`N2g-per-comp`)
- **Random-circuit simulator** with even/odd brick-wall two-qubit layers
- Lightweight TTN + circuit backend and **layer-wise fidelity tracking**
- **Reproducible graphs** (linear chain, 3-regular)
- **Results** automatically logged under `results/`

## Installation

### Directly from GitHub
```bash
pip install "git+https://github.com/ZekiZeybek/SimQC-TTN.git"
```

### From a local clone 
```bash
# run from the repo root (the folder with pyproject.toml; not inside src/)
pip install .
```

### Developer mode 
```bash
# Useful for modifying the code as you develop and check ...
# meaning you want to work on the code and see your changes take effect right away, without reinstalling the package each time
pip install -e "."
```

## Usage

### Quickstart

```python
# random_circuit_demo.py
# If running without installation:
#   python random_circuit_demo.py   (from the repo root)

import sys, os
sys.path.insert(0, os.path.abspath("src")) 

from simqc_ttn.simulation.sim_rand_circ import sim_rand, chi_bra_layers
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Configuration --- #
num_qubits   = 16
circ_depth   = 4                 # number of (odd+even) brick-wall blocks
chi_bra      = chi_bra_layers([12, 8, 4, 2])  # TTN bond dims per tree layer, see the notebooks for how this maps to binary tree layers
num_sweeps   = 4                 # DMRG sweeps per compression

# --- Run --- #
final_state, layer_fids = sim_rand(
    circ_depth=circ_depth,
    num_of_qbts=num_qubits,
    chi_bra=chi_bra,
    num_sweeps=num_sweeps,
)

F_tilde = np.prod(layer_fids)
print("Total simulation fidelity (product over layers):", F_tilde)

# --- Plot ---- 
plt.figure()
plt.plot(np.arange(1, len(layer_fids)+1), layer_fids, marker="o")
plt.xlabel("Depth")
plt.ylabel("Fidelity")
plt.title("Random-circuit compression fidelities per depth")
plt.show()
```

### Command line

```bash
# Flexible QAOA on a 3-regular graph, 16 qubits, 2 layers
simqc-ttn --out 3reg_qaoa_output qaoa --mode flex --num_qbts 16 --layers 2 --default_graph reg3 --graph_seed 2 --chi 32,16,4,2 --N2g-per-comp 12 --sweeps 4

# Flexible QAOA on a linear chain, 16 qubits, 2 layers
simqc-ttn --out lin_qaoa_output qaoa --mode flex --num_qbts 16 --layers 2 --default_graph linear --chi 32,16,4,2 --N2g-per-comp 12 --sweeps 4

# User-defined edges
simqc-ttn --out user_edges_qaoa qaoa --mode flex --num_qbts 16 --layers 2 --edges "0-4,1-5,2-6" --chi 32,16,4,2 --N2g-per-comp 2 --sweeps 4

# Native QAOA on a 3-regular graph
simqc-ttn --out reg3_qaoa_native qaoa --mode native --num_qbts 16 --layers 2 --default_graph reg3 --chi 32,16,4,2 --sweeps 4

# Random circuit
simqc-ttn --out rnd_circ_output rand --num_qbts 16 --layers 2 --chi 16,8,4,2 --sweeps 2
```
> **Terminology:**  
> • For **QAOA**, `--layers` = number of QAOA layers  
> • For **Random**, `--layers` = number of (even+odd) two-qubit blocks (“depth”) 

### Notebooks
See `notebooks/`:
- `example_qaoa.ipynb` — native and flexible QAOA 
- `example_random.ipynb` — random-circuit demo  

## Limitations & Scope
- **Models**: QAOA (Ising ZZ cost) and random circuits; other cost Hamiltonians not yet implemented
- **Backends**: Binary TTN only
- **QAOA**: QAOA parameters and edge weights are all set to unity

## TODO
- Used defined QAOA parameters and edge weights
- Optimization backend for QAOA angles and cost function calculation with the resulting state
- General balanced/unbalanced tree tensor network backend
- SVD-based compression algorithm for initializing the variational state
## Notes

For a more generalized codebase generating published results, see **[Quantum-Circuit-Simulator-DMRG](https://github.com/AdityaD16/Quantum-Circuit-Simulator-DMRG)**, used in:  
“Simulating Quantum Circuits with Tree Tensor Networks using Density-Matrix Renormalization Group Algorithm,” *Phys. Rev. B* 112, 104312 (2025)

## Reference

Aditya Dubey, Zeki Zeybek, and Peter Schmelcher,  
**“Simulating Quantum Circuits with Tree Tensor Networks using Density-Matrix Renormalization Group Algorithm,”**  
[Phys. Rev. B 112, 104312 (2025)](https://journals.aps.org/prb/abstract/10.1103/64hd-q4z5)  
