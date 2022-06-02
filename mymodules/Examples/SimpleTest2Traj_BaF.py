# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:35:14 2022

@author: fkogel

tested with version v3.0.0
"""

from System import *
system = System(description='SimpleTest2Traj_BaF',load_constants='BaFconstants')

# specify initial velocity and position of the molecule
system.v0 = np.array([200,0,0])   #in m/s
system.r0 = np.array([-2e-3,0,0]) #in m

# set up the cooling laser and first repumper with their wave vectors k and positions r_k
FWHM,P = 1e-3,5e-3 # 1mm and 5mW
for lamb in np.array([859.830, 895.699])*1e-9:
    for rx in [0, 4e-3]:
        system.lasers.add_sidebands(lamb=lamb,P=P,FWHM=FWHM,pol='lin',
                                    r_k=[rx,0,0], k=[0,1,0],
                                    offset_freq=19e6,mod_freq=39.33e6,
                                    sidebands=[-2,-1,1,2],ratios=[0.8,1,1,0.8])
    
# include first two vibrational levels of electronic ground state and the
# first vibrational level of the excited state
system.levels.add_all_levels(v_max=1)

#%% calculate dynamics with velocity and position dependence of the laser beams and molecules
system.calc_rateeqs(t_int=40e-6,magn_remixing=False,
                    trajectory=True,position_dep=True)

# plot scattering rate, scattered photons, velocity and position, ...
system.plot_all()
