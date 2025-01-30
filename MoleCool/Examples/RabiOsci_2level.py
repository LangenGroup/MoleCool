# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:44:02 2023

@author: fkogel

tested with version v3.2.2
"""
from MoleCool import System, pi, plt
 
system = System('Rabi-2level')

system.levels.add_electronicstate('g', 'gs') # ground electronic state
system.levels.g.add(F=0,mF=0) # add a single level with F=0

system.levels.add_electronicstate('e', 'exs') # excited electronic state
system.levels.e.add(F=1,mF=0) # add single mF=0 with F=1 as F=0 would be forbidden

system.levels.g.set_init_pops({'F=0':1.0}) # initial population

ratio_OmGa  = 20 # ratio between Rabi frequency and the linewidth
Omega       = system.levels.calc_Gamma()[0] * ratio_OmGa # Rabi frequency
T_Om        = 2*pi/Omega # time of one period

plt.figure(system.description)
plt.ylim([0,1])
plt.xlabel('Time $t$ [$2\pi/\Omega$]')
plt.ylabel('Excited state population $n^e$')
for det in [0,1,2]:
    del system.lasers[:] # delete laser instances in every iteration
    system.lasers.add(freq_shift = det*Omega/2/pi, freq_Rabi = Omega) # add laser component
    # or alternatively (using intensity instead of directly providing the Rabi freq.):
    # system.lasers.add(freq_shift = det*Omega/2/pi,
                      # I = 2*system.levels.Isat[0,0]*ratio_OmGa**2)
    
    system.calc_OBEs(t_int=5*T_Om, dt=1e-2*T_Om) # calculate dynamics with OBEs
    plt.plot(system.t/T_Om, system.N[1,:], label=str(det))
    
plt.legend(title='$\Delta/\Omega$',loc='upper right',ncols=3)
plt.savefig("Fig2_Rabi-2level")