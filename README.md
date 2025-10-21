# MoleCool

<p align="center">
  <img src="doc/source/_static/molecool_logo.png" alt="MoleCool logo" width="200"/>
</p>

**MoleCool** is a Python library for **numerical modeling of light–matter interactions**,  
with a focus on **laser cooling of molecules** — from simple few-level systems and diatomics  
to polyatomics and even radioactive species.

<p align="center">
  <a href="https://molecool-py.readthedocs.io/"><b>📘 Full Documentation</b></a> •
  <a href="https://pypi.org/project/MoleCool/"><b>PyPI</b></a> •
  <a href="https://github.com/LangenGroup/MoleCool/"><b>GitHub</b></a>
</p>

---

## ✨ Overview

**MoleCool** enables you to **simulate, analyze, and visualize** the dynamics of laser–molecule interactions  
for designing and optimizing experimental setups in state-of-the-art cooling and trapping experiments.

### Key Features

- **Flexible dynamics solvers** — rate equations and Optical Bloch equations (OBEs),  
  including effects of external magnetic fields.  
  MoleCool reads predefined molecular constants (dipole matrix elements, hyperfine frequencies,  
  g-factors, etc.) from JSON files.

- **Interactive level-scheme handling** — intuitive tools for exploring and visualizing  
  electronic, vibrational, and rotational structures.

- **Laser cooling force profiles** — fast evaluation across high-dimensional parameter spaces  
  for optimization of cooling configurations.

- **Monte Carlo trajectory simulations** — track many particles through laser fields  
  using pre-evaluated force profiles for statistical reliability.

- **Spectra analysis module** — compute, fit, and interpret molecular spectra via  
  effective Hamiltonians; extract constants for use in dynamics simulations.

### Target Audience

MoleCool is intended for **researchers and physicists** working in  
**atomic, molecular, and optical (AMO) physics**, particularly in:

- Modeling laser–molecule interactions  
- Optimizing cooling and trapping experiments  
- Interpreting or predicting spectroscopic measurements  

---

## ⚙️ Installation

We recommend installing MoleCool in a dedicated virtual environment  
(using either `virtualenv` or `conda`) to avoid dependency conflicts.
See the [Installation Guide](https://molecool-py.readthedocs.io/en/stable/installation.html)
for more details.

> **Requires:** Python ≥ 3.8 (Python ≤ 3.10 recommended)

### Using `pip` (stable release)

```bash
pip install MoleCool
```

### Using `conda` (via conda-forge)

```bash
conda install -c conda-forge MoleCool
```

### Development version (latest from GitHub)

```bash
git clone https://github.com/LangenGroup/MoleCool
cd MoleCool
pip install .
```

---

## 🧪 Verifying the Installation

You can verify a correct installation by running the built-in example suite:

```bash
python -m MoleCool.run_examples
```

Add the `-h` flag for a help message and a list of all available example scripts.

---

## 🚀 Quickstart Example

Below is a minimal working example demonstrating MoleCool’s basic workflow:

```python
from MoleCool import System

# Initialize a molecular system (e.g. 138BaF)
system = System(load_constants='138BaF')

# Build level scheme and remove loss channels
system.levels.add_all_levels(v_max=0)
system.levels.X.del_lossstate()

# Define a multi-sideband laser configuration
system.lasers.add_sidebands(
    lamb        = 859.83e-9,
    P           = 20e-3,
    offset_freq = 19e6,
    mod_freq    = 39.33e6,
    sidebands   = [-2, -1, 1, 2],
    ratios      = [0.8, 1, 1, 0.8],
)

# Turn on a magnetic field
system.Bfield.turnon(strength=5e-4, direction=[1, 1, 1])

# Run dynamics simulations
system.calc_OBEs(t_int=8e-6, dt=1e-9, magn_remixing=True)
system.calc_rateeqs(t_int=8e-6, magn_remixing=True, position_dep=True)

# Visualize populations
system.plot_N()
```

For detailed usage, see the  
👉 [User Guide](https://molecool-py.readthedocs.io/en/stable/user_guide/introduction.html)  
and [Examples](https://molecool-py.readthedocs.io/en/stable/examples.html).

---

## ⚡ Performance and Parallelization

- Dynamics equations are solved using [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
  and compiled with **Numba**’s just-in-time (JIT) compiler for near-C speed.  
- Independent simulations are automatically parallelized across multiple cores  
  using Python’s `multiprocessing` module — ideal for large parameter sweeps  
  and Monte Carlo trajectory studies.

---

## 🤝 Contributing

Contributions are welcome! To develop MoleCool locally:

```bash
git clone https://github.com/LangenGroup/MoleCool
cd MoleCool
pip install -e .[dev,doc]
```

The `-e` flag enables **editable mode**, allowing immediate testing of code changes.  
See the [Installation Guide](https://molecool-py.readthedocs.io/en/stable/installation.html)
for more details.

> ⚠️ **Important:** Do not import the package from its parent folder if the
> directory name is also `MoleCool`, as this can confuse Python’s import system.

---

## 📚 Documentation

Full documentation, tutorials, and API reference are hosted on **ReadTheDocs**:

👉 [https://molecool-py.readthedocs.io/](https://molecool-py.readthedocs.io/)

---

## 🧾 Citation

If you use MoleCool in your research, please cite the corresponding paper:  
[![arXiv](https://img.shields.io/badge/arXiv-2510.16203-blue.svg)](https://arxiv.org/abs/2510.16203)

---

## 🧠 License

© Felix Kogel — released under the **MIT License**.  
See [`LICENSE`](./LICENSE.md) for details.

---

### 👩‍🔬 Developed by Felix Kogel
For questions or feedback, open an issue on [GitHub](https://github.com/LangenGroup/MoleCool/issues).

---

**MoleCool** — *A modular Python framework for simulating laser cooling of molecules.*
