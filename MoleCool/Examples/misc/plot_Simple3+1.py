# -*- coding: utf-8 -*-
"""
Simplest type-II levelsystem
============================

creating 3+1 level system and observing time-dependent populatios.
"""

from MoleCool import System

system = System(description='Simple3+1') # create empty system instance first

# construct level system:
# - create empty instances for a ground and excited electronic state
system.levels.add_electronicstate(label='X', gs_exs='gs')
system.levels.add_electronicstate(label='A', gs_exs='exs', Gamma=1.0)
# - add the levels with the respective quantum numbers to the electronic states
system.levels.X.add(F=1)
system.levels.A.add(F=0)
# - next all default level properties can be displayed and simply changed
system.levels.print_properties()
system.levels.X.gfac.iloc[0] = 1.0 # set ground state g factor to 1.0

# set up lasers and magnetic field
system.lasers.add(lamb=860e-9, P=5e-3, pol='lin') #wavelength, power, and polarization
system.Bfield.turnon(strength=5e-4, direction=[0,1,1]) #magnetic field

# simulate dynamics with OBEs and plot population
system.calc_OBEs(t_int=5e-6, dt=10e-9, magn_remixing=True, verbose=True)
system.plot_N()