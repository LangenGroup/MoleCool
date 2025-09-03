Getting Started
===============

Installation
------------
.. code-block:: bash

   pip install molecool

Quickstart
----------
.. code-block:: python

   import molecool as mc
   mol = mc.Molecule("H2O")
   mol.simulate(steps=1000)
   mol.visualize()
