# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:35:14 2022

@author: fkogel

tested with version v3.1.0
"""

from MoleCool import System, np
system = System(description='SimpleTest1_BaF',load_constants='138BaF')

# set up the lasers each with four sidebands
for lamb in np.array([859.830, 895.699, 897.961])*1e-9:
    system.lasers.add_sidebands(lamb=lamb,P=20e-3,pol='lin',
                                offset_freq=19e6,mod_freq=39.33e6,
                                sidebands=[-2,-1,1,2],ratios=[0.8,1,1,0.8])

# set up the ground and excited states and include all vibrational levels
# up to the ground state vibrational level v=2
system.levels.add_electronicstate('X','gs') #add ground state X
system.levels.X.load_states(v=[0,1,2]) #loading the states defined in the json file
system.levels.X.add_lossstate()

system.levels.add_electronicstate('A','exs') #add excited state A
system.levels.A.load_states(v=[0,1])

system.levels.print_properties() #check the imported properties from the json file

# Alternatively, all these steps for the levels can be done simpler with:
# system.levels.add_all_levels(v_max=2)

#%%
# calculate dynamics with rate equations
system.calc_rateeqs(t_int=20e-6, magn_remixing=False)

# plot populations, force, and scattered photon number
system.plot_N()
system.plot_F()
system.plot_Nscatt()
