Introduction
============

MoleCool is a Python library for numerical modeling of light-matter interactions,
specializing in laser cooling in molecules -- from simple few-level systems
and diatomics to polyatomics and radioactive species.

With MoleCool, you can simulate, analyze, and visualize the dynamics of laser-molecule interactions 
to design and optimize experimental setups for cutting-edge cooling and trapping experiments.


Key Features
------------

- **Flexible dynamics solvers**  
  Supports rate equations and Optical Bloch equations (OBEs) to evaluate interactions in the 
  presence of external magnetic fields. MoleCool reads predefined constants such as dipole matrix 
  elements, hyperfine frequencies, and g-factors from JSON files.

- **Interactive level scheme handling**  
  Intuitive tools to explore and adjust electronic, vibrational, and rotational level structures.  
  MoleCool can automatically generate clear visualizations of:
  
  - Intricate level schemes
  - Transition spectra
  - Time-dependent light-matter dynamics

- **Laser cooling force profiles**  
  Efficient evaluation of cooling force profiles across high-dimensional parameter spaces, 
  enabling fast optimization for complex cooling configurations.

- **Monte Carlo trajectory simulations**  
  Simulate the motion of many individual particles through laser fields using pre-evaluated 
  force profiles for statistically meaningful results.

- **Spectra analysis module**  
  An independent ``spectra`` module allows analysis and fitting of molecular spectra 
  using an effective Hamiltonian. This makes it possible to extract spectroscopic constants 
  (dipole matrix elements, hyperfine frequencies, g-factors, ...), which can then be fed back 
  into the dynamics simulations.

Target Audience
---------------

MoleCool is intended for **researchers and physicists** in the field of **atomic and molecular laser cooling**, 
who require a versatile simulation framework to:

- Model laser-molecule interactions
- Optimize experimental configurations
- Interpret or predict spectroscopic measurements

Quick Example
-------------

Below is a minimal example of setting up a molecular system, adding lasers, 
switching on a magnetic field, and running both OBE and rate equation simulations:

.. code-block:: python

    from MoleCool import System
    
    system = System(load_constants='138BaF')
    
    # Build level scheme
    system.levels.add_all_levels(v_max=0)
    system.levels.X.del_lossstate()
    
    # Add laser configuration
    system.lasers.add_sidebands(
        lamb        = 859.83e-9,
        P           = 20e-3,
        offset_freq = 19e6,
        mod_freq    = 39.33e6,
        sidebands   = [-2, -1, 1, 2],
        ratios      = [0.8, 1, 1, 0.8]
    )
    
    # Magnetic field
    system.Bfield.turnon(strength=5e-4, direction=[1, 1, 1])
    
    # Dynamics simulations
    system.calc_OBEs(t_int=8e-6, dt=1e-9,
                     magn_remixing=True)
    system.calc_rateeqs(t_int=8e-6, magn_remixing=True,
                        position_dep=True)
    system.plot_N()

For further details, see the :doc:`Installation <../installation>` 
and :doc:`Getting started <getting_started>` sections.