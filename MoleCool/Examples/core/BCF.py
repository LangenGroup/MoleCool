# -*- coding: utf-8 -*-
"""
Bichromatic force
=================

Calculating the bichromatic force and compare BaF with CaF.

Steady state coherent force BCF...
"""
from MoleCool import System, np, plt, pi, c, open_object, save_object
import os.path

# %%
# Calculation for BaF and CaF
# ---------------------------
if __name__ == '__main__': 
    for label, exAB in zip(['138BaF','CaF'], ['A','B']):
        if not os.path.isfile(f"BCF_{label}.pkl"):
            
            system = System(load_constants  = label,
                            description     = f"BCF_{label}")
            
# %%
# Electronic states and velocity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            system.levels.add_electronicstate('X', 'gs')
            system.levels.X.load_states(v=[0])
            system.levels.add_electronicstate(exAB, 'exs')
            system.levels[exAB].load_states(v=[0])
    
            system.set_v0(
                np.arange(-120.25, 120.50, 0.5),
                direction = 'z',
                )
    
# %%
# Lasers with right Rabi frequency
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
            Gamma       = system.levels.calc_Gamma()[0]
            delta       = 75*Gamma /(2*pi)
            chi         = pi/4
            det         = {'138BaF':20e6, 'CaF':-38e6}[label]
            Isat        = system.levels.Isat.mean() # = pi*c*h*Gamma/(3*531e-9**3)
            I           = 3*Isat*(delta*2*pi/(Gamma/np.sqrt(3)))**2
            
            for s1 in [+1,-1]:
                for s2 in [+1,-1]:
                    system.lasers.add(
                        lamb            = system.levels.wavelengths.loc[('X',0),(exAB,0)]*1e-9,
                        pol             = 'lin',
                        pol_direction   = 'z',
                        I               = I,#freq_Rabi=np.sqrt(3/2)*delta*2*pi,
                        freq_shift      = s2*delta+det,
                        k               = [0,0,s1*1],
                        phi             = -1*s1*s2*chi/2
                        )
            
# %%
# Magnetic field
# ^^^^^^^^^^^^^^

            system.Bfield.turnon(
                strength    = {'138BaF' : 20e-4, 'CaF' : 33e-4}[label],
                angle       = {'138BaF' : 60,    'CaF' : 63.5}[label],
                direction   = np.array([0.,0.,1.]),
                )
            
# %%
# OBEs
# ^^^^
            system.steadystate['t_ini'] = 2e-6
            system.steadystate['period'] = 'standingwave'
            system.steadystate['maxiters'] = 8
            system.steadystate['condition'] = [0.2,10]
            # system.multiprocessing['maxtasksperchild'] = None
            # system.multiprocessing['processes'] = 20
            system.multiprocessing['savetofile'] = True
            
            system.calc_OBEs(
                t_int           = 4e-6,
                dt              = 0.2e-9,
                method          = 'RK45',
                magn_remixing   = True,
                verbose         = False,
                steadystate     = True,
                rounded         = False,
                freq_clip_TH    = 1000,
                )
        

# %%
# Loading data and plotting
# -------------------------

    plt.figure()
    
    for label in ['138BaF','CaF']:
        system  = open_object(f'BCF_{label}')
        F       = system.results.vals['F']/system.hbarkG2
        v0      = system.v0[:,2]

        plt.plot(v0, F[:,2], label = label)
        
    plt.xlabel('Velocity $v$ (m/s)')
    plt.ylabel('Bichromatic force $F_{BCF}$ ($\hbar k \Gamma/2$)')
    plt.legend()
    
# %%
# .. image:: /_figures/core/BCF_fig1.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/BCF_fig1.svg'