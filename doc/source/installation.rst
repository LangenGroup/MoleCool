Installation
============

We recommend installing **MoleCool** inside a dedicated virtual environment
to avoid dependency conflicts.
You can create a virtual environment with e.g. the popular tool
`virtualenv <https://pypi.python.org/pypi/virtualenv>`_
or with ``conda``.

.. note::
   The module requires at least ``python 3.8``.
   However, ``python <=3.10`` is recommended.

Creating a virtual environment
------------------------------

.. tab-set::

    .. tab-item:: Linux / macOS

        .. code-block:: bash

            # Create a virtual environment
            virtualenv -p python3.10 .venv

            # Activate the virtual environment
            source .venv/bin/activate

    .. tab-item:: Windows

        .. code-block:: powershell

            # Create a virtual environment
            python -m virtualenv -p python3.10 .venv

            # Activate the virtual environment using:
            # - Command Prompt / cmd.exe
            .\.venv\Scripts\activate.bat
            
            # - PowerShell
            .\.venv\Scripts\Activate.ps1

    .. tab-item:: Conda (all platforms)

        .. code-block:: bash

            # Create a new conda environment with Python 3.10 (recommended)
            conda create -n venv python=3.10

            # Activate the environment
            conda activate venv


Installing MoleCool
-------------------

Once your virtual environment is active (or your conda env is activated),
install **MoleCool** using one of the following methods.

.. tab-set::

    .. tab-item:: pip (stable)

        The easiest way is to install the latest stable release from PyPI.
        This will install the version that has been officially released and tested.
        
        .. code-block:: bash

            python -m pip install --upgrade pip
            pip install MoleCool

    .. tab-item:: conda (stable)

        Using conda a stable release via conda-forge can be installed via:
        
        .. code-block:: bash

            conda install -c conda-forge MoleCool

    .. tab-item:: git + pip (beta)

        This requires ``git`` to be installed which is used to download
        the latest development version of the module from GitHub initiating
        a local copy of the repository.
        
        .. code-block:: bash

            git clone https://www.github.com/LangenGroup/MoleCool
            cd MoleCool
            python -m pip install --upgrade pip
            pip install .


Contributing
------------

.. dropdown:: How to contribute in the package's development

    .. button-link:: https://github.com/LangenGroup/MoleCool/
       :color: secondary
       :outline:

       :fab:`github` GitHub
       
    If you want to contribute to the code development hosted on GitHub,
    manually clone the latest development version using git (see above) and
    install the module along with optional dependencies for devopment or
    documentation by using the ``dev`` and ``doc`` extras.
    
    .. code-block:: bash

        pip install -e .[dev,doc]    

    Adding the ``-e`` flag enables editable mode, which allows you to modify
    the source code and immediately test changes without repeatedly
    running ``pip install .``.
    
    .. important::
        Do not import the package (``import MoleCool``) from the repository's
        parent folder (as current working directory)
        if it has the same name as the package (``MoleCool``).
        Doing so can confuse Python, as it may mistake the repository folder
        for the package itself.
    

Verifying the installation
--------------------------

To ensure that MoleCool has been installed correctly, run the provided example suite:

.. code-block:: bash

    python -m MoleCool.run_examples

This will run a set of fast example scripts included with the package and verify
that your installation is working correctly.

.. note::

   By adding the flag ``-h``, you can display the help message along with a list
   of all runnable examples.
   Additionally, there are longer example scripts designed to be run on a
   compute (HPC) server for optimal performance.
   These scripts also allow you to specify whether the generated plots should
   be saved to files or displayed directly.
   
   All available example scripts are presented in a well-organized, readable,
   and documented format in the :doc:`Examples <examples>` section.

Quickstart
----------
.. code-block:: python

   from MoleCool import System
   
   system = System(load_constants='138BaF')
   system.levels.add_all_levels(v_max=0)
   
   system.levels.print_properties()
