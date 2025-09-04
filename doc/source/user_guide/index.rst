User Guide
==========

This guide will walk you through everything you need to know to get started with MoleCool.

.. exec_code::

   import numpy as np
   np.arange(5)
   
.. toctree::
   :maxdepth: 2

   getting_started
   introduction
   quickstart
   
..
    Information
    -----------
    This python software includes two different programs:
    
    1. Molecular Dynamics Simulation Code
    	The first code only uses pre-defined constants (like the dipole matrix,
    	hyperfine freqs, g-factors, ...) from a .json file which are required
    	to calculate dynamics via the rate or optical Bloch equations.
    	
    2. Molecular Spectra Calculation Code
    	In comparison, the second code only uses constants of the effective
    	Hamiltonian and Quantum numbers of the level structure to evaluate the
    	constants (like dipole matrix, hyperfine freqs, g-factors, ...) which
    	are needed for the dynamics simulation code.
    
    --------------------------
    
    To get started:
    ^^^^^^^^^^^^^^^
    
    + SVN Checkout of the repository
    + the first command in python3 should be::
    	
    	>>> from System import *
    	
    + This command should be called from the **same directory** as where the
      modules are stored. Otherwise the module's directory can also be included
      in an environment variable named "PYTHONPATH" in Windows to call the
      command from an arbitrary working directory
    + Then, you see if some further packages have to be installed (e.g. numba, pandas, ..)
    + Afterwards, some examples from the module's documentation can be tried.
    
    ----------------------------
    
    
    Code Structure:
    ^^^^^^^^^^^^^^^
    
    .. figure:: CodeStructure.jpg
       :scale: 70
       :align: center
       :alt: map to buried treasure
       :figclass: align-center
       
       Class diagram of the ``Molecular Dynamics Simulation Code``. While methods
       are characterized by parentheses,
       attributes and properties are labeled without parentheses.
       Class composition is marked by arrows with diamonds and class inheritance by open arrow tip.
    
    
    .. toctree::
       :maxdepth: 2
       :caption: API documentation
       
       modules.rst
    
    ------------------------
    
    Indices and tables
    ==================
    
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
