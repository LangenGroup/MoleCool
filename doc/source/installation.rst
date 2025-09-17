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
            python -m virtualenv -p python3.8 .venv

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

Once your virtual environment is active (or your conda env is activated), install **MoleCool** using one of the following methods:

.. tab-set::

    .. tab-item:: pip (from PyPI)

        .. code-block:: bash

            pip install MoleCool

    .. tab-item:: conda (conda-forge)

        .. code-block:: bash

            conda install -c conda-forge MoleCool

    .. tab-item:: Manual (local repository folder)

        .. code-block:: bash

            # From a local wheel or source archive
            pip install /path/to/downloaded/repository


Contributing
------------

.. dropdown:: How to contribute in the package's development

    .. button-link:: https://github.com/LangenGroup/MoleCool/
       :color: secondary
       :outline:

       :fab:`github` GitHub
       
    To contribute to the code development hosted on GitHub,
    additional modules need to be installed, which can be achieved
    by adding the devopment (dev) or documentation (doc) labels:
    
    .. code-block:: bash

        pip install -e MoleCool[dev,doc]
    

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
