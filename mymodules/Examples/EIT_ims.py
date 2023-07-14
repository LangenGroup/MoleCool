# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:09:30 2023

@author: fkogel

tested with version v3.1.1
"""
from System import *

det_pump = +1*1e6 # pump laser detuning for for EIT resonance position in Hz
I_probe  = 0.2 # laser intensity of probe beam in W/m^2
Gamma_r  = 1e-3
I_pump   = 5.0/Gamma_r # laser intensity of pump beam in W/m^2
gr_split = 1000 # splitting between both ground states in MHz
#%%
Deltas  = np.array([*np.arange(-4,-0.5,0.1),*np.arange(-0.5,0.,0.01)])*1e6
Deltas  = np.array([*Deltas,*(-np.flip(Deltas[:-1]))]) + det_pump

rho     = np.zeros((2,len(Deltas)),dtype=complex) # steady state density matrix elements

for i in range(0,2):
    system = System('testingEIT')
    system.steadystate['condition'] = [0.01,0.1]
    system.levels.add_electronicstate('e', 'ims') # first add only excited state
    system.levels.e.add(F=1,mF=0) # add single mF=0 with F=1 as F=0 would be forbidden

    # simple two-level case:
    if i == 0:
        system.levels.add_electronicstate('g', 'gs') # ground electronic state
        system.levels.g.add(J=0.5,F=0,mF=0) # add a single level with F=0
        # initial population can be specified but makes no difference for steady state
        system.N0 = [1,0,0]
        
    # Lambda-level scheme for EIT case:
    if i == 1:
        system.levels.add_electronicstate('g', 'gs') # ground electronic state
        system.levels.g.add(J=0.5,F=0,mF=0) # add two single levels with F=0
        system.levels.add_electronicstate('r', 'exs',Gamma=Gamma_r) # ground electronic state
        system.levels.r.add(J=1.5,F=0,mF=0)
        system.levels.r.freq.iloc[0] = +gr_split # detuning in MHz between both ground states
        system.levels.transdipmoms.iloc[0,0] = 0.0
        # system.levels.transdipmoms.iloc[1,0]*=-1
        system.levels.get_wavelengths('g','r').iloc[0,0] = 500.
        I_probe /=  2 # that Rabi freqs of probe beam is exactly the same
        I_pump /= 2
    
    # Iterating over all detunings of scanning probe laser:
    for j,Delta in enumerate(Deltas):
        del system.lasers[:] # delete laser instances in every iteration
        system.lasers.add(I=I_probe, freq_shift=Delta) # (weak) p<robe laser
        # only add pump laser for EIT case:
        if i == 1:
            system.lasers.add(I=I_pump, freq_shift=+gr_split*1e6-det_pump)
        
        T_Om = 2*pi/system.calc_Rabi_freqs().max() # period time of Rabi frequency
        # calculate dynamics until steady state is reached
        system.calc_OBEs(t_int=5*T_Om, steadystate=True, verbose=False)
        
        # transform density matrix elements (ymat) to the other rotating frame
        # which rotates with level's transition frequency instead laser frequency
        rho[i,j] = (system.ymat[0,system.levels.N-system.levels.iNum,:] * np.exp(1j*system.t*Delta*2*pi)).mean()

#%% plotting
plt.figure('Susceptibility new')
for i,case in enumerate(['2-levels','$\Lambda$-EIT']):
    plt.plot(Deltas*1e-6,rho[i].imag,'--',label='$Im(\\rho_{eg})$, '+case, c='C'+str(i))
    plt.plot(Deltas*1e-6,rho[i].real,'-',label='$Re(\\rho_{eg})$, '+case, c='C'+str(i))
plt.xlabel('Probe laser detuning [MHz]')
plt.ylabel('Density matrix element $\\rho_{eg}$')
plt.legend()

