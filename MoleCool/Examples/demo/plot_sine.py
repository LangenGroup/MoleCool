"""
Plotting a sine wave
====================

This tutorial shows how to create and plot a simple sine wave.

The top-level docstring becomes the intro text.
"""

# %%
# Step 1 — imports
# ----------------
import numpy as np
import matplotlib.pyplot as plt

# %%
# Step 2 — generate data
# ----------------------
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

# %%
# Step 3 — plot
# -------------
plt.plot(x, y, label="sin(x)")
plt.title("Simple sine wave")
plt.legend()

# %%
# This is a section header
# ------------------------
#
# In the built documentation, it will be rendered as reST. All reST lines
# must begin with '# ' (note the space) including underlines below section
# headers.

# These lines won't be rendered as reST because there is a gap after the last
# commented reST block. Instead, they'll resolve as regular Python comments.
# Normal Python code can follow these comments.
myvariable = 2
print('my variable plus 2 is {}'.format(myvariable + 2))

# %%
# Levels test
# -----------

from MoleCool import Levelsystem
levels = Levelsystem()
levels.add_electronicstate('S12', 'gs')     # define as ground state ('gs')
levels.S12.add(J=1/2,F=[1,2])
state1 = levels.S12[0]
print(state1)

# %%
# MoleCool test
# -------------
# Use :class:`MoleCool.System.System` class.

from MoleCool import System, np
from MoleCool import Bfield as B
from MoleCool.tools import get_constants_dict
print(B(), get_constants_dict())
system = System(description='SimpleTest1_BaF',load_constants='138BaF')

system.levels.add_electronicstate('X','gs') #add ground state X
system.levels.X.load_states(v=[0,1]) #loading the states defined in the json file
system.levels.X.add_lossstate()

system.levels.add_electronicstate('A','exs') #add excited state A
system.levels.A.load_states(v=[0])

system.levels.print_properties() #check the imported properties from the json file

# %%
# test directly pandas dataframe

from MoleCool.System import System
s2 = System()
system.levels.dMat_red

#%%

from MoleCool.spectra import Molecule
Molecule()
