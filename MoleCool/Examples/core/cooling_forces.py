# -*- coding: utf-8 -*-
"""
Cooling forces
==============

Calculating the transversal cooling forces for BaF and CaF.

Steady state cooling forces...
"""
from MoleCool.tools import plot_results, get_results
from MoleCool import System, np, plt, pi, c, open_object
import os.path

from scipy.special import jv
def compute_eom_sidebands(modulation_index, max_order=10):
    orders      = np.arange(-max_order, max_order + 1)
    intensities = jv(orders, modulation_index)**2
    intensities /= np.sum(intensities)
    return orders, intensities

if __name__ == '__main__':    
    if not os.path.isfile("cooling_forces_CaF.pkl"):
    
# %%
# CaF cooling
# -----------
#
# Initialize levels
# ^^^^^^^^^^^^^^^^^
# 
# We load the predefined constants of :math:`^{138}\text{BaF}` and :math:`\text{CaF}`
# (similarly to :ref:`sphx_glr_auto_examples_core_optcycl_12+4levels.py`).

        system = System(load_constants  = 'CaF',
                        description     = 'cooling_forces_CaF')
        
        # set up electronic states
        system.levels.add_electronicstate('X','gs')
        system.levels.X.load_states(v=[0])  
        system.levels.add_electronicstate('B','exs')
        system.levels.B.load_states(v=[0])
        
# %%
# Velocity and magnetic field
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        system.set_v0([0, 0.05, 0])

        system.Bfield.turnon(
            strength    = 0.2e-4,
            direction   = np.array([0.,0.,1.]),
            )
        
# %%
# Laser system
# ^^^^^^^^^^^^
        
        detuning = np.arange(-18, 22, 0.3333) * system.levels.B.Gamma *1e6
        EOM_sideband_orders, EOM_sideband_ratios = compute_eom_sidebands(1.84, 3)
        
        for k_dir in [[0, 1, 0], [0, -1, 0]]:
            system.lasers.add_sidebands(
                lamb            = system.levels.wavelengths.loc[('X',0),('B',0)]*1e-9,
                FWHM            = 4e-3,
                P               = 50e-3,
                offset_freq     = detuning, #22.3e6 + detuning,
                mod_freq        = 1e6,
                sidebands       = 72 * EOM_sideband_orders,
                ratios          = EOM_sideband_ratios,
                k               = k_dir,  # y-direction laser
                pol_direction   = [1, 0, 1] # pol_direction
            )
        
# %%
# OBEs evaluation
# ^^^^^^^^^^^^^^^
        system.steadystate.update(dict(
            t_ini       = 5e-6,
            period      = 'standingwave',
            maxiters    = 10,
            condition   = [0.05,1]
            ))
        
        system.multiprocessing.update(dict(
            maxtasksperchild    = 5,
            show_progressbar    = True,
            savetofile          = True,
            # processes           = 30,#v0_arr.shape[0],
            random_evaluation   = False,
            ))
        
        system.calc_OBEs(
            # t_int           = 30e-6, #30.4e-6
            dt              = 'auto',#1e-9,
            method          = 'RK45',
            #rtol=1e-4,atol=1e-6,
            magn_remixing   = True,
            verbose         = True,
            steadystate     = True,
            rounded         = False,
            freq_clip_TH    = 500,
            )
# %%
# BaF cooling
# -----------
#
# Initialize levels
# ^^^^^^^^^^^^^^^^^
    if not os.path.isfile("cooling_forces_BaF.pkl"):
        
        system = System(load_constants  = '138BaF',
                        description     = 'cooling_forces_BaF')
        
        # set up electronic states
        system.levels.add_electronicstate('X','gs')
        system.levels.X.load_states(v=[0])  
        system.levels.add_electronicstate('A','exs')
        system.levels.A.load_states(v=[0])
    
# %%
# Velocity and magnetic field
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

        system.set_v0([0, 0, 0.05])
        
        system.Bfield.turnon(
            strength    = 0.5e-4,
            angle       = 45,
            direction   = np.array([0.,0.,1.]),
            )
        
# %%
# Laser system
# ^^^^^^^^^^^^

        detuning = np.arange(-20,20,0.3333)
        
        EOM_sideband_orders, EOM_sideband_ratios = compute_eom_sidebands(2.5, 3)
        for k in [[0.,0.,1.],[0.,0.,-1.]]:
            system.lasers.add_sidebands(
                lamb        = system.levels.wavelengths.loc[('X',0),('A',0)]*1e-9,
                I           = 5000,
                # I           = 500.,
                pol         = 'lin',
                k           = k,
                mod_freq    = 39.3e6,
                ratios      = EOM_sideband_ratios,
                sidebands   = EOM_sideband_orders,
                offset_freq = detuning*1e6*system.levels.A.Gamma + 20e6,
                )
        
# %%
# OBEs evaluation
# ^^^^^^^^^^^^^^^
        system.steadystate.update(dict(
            t_ini       = 20e-6,
            period      = 'standingwave',
            maxiters    = 10,
            condition   = [0.05,1], #[0.1,10]
            ))
        
        system.multiprocessing.update(dict(
            maxtasksperchild    = 5,
            show_progressbar    = True,
            savetofile          = True,
            # processes           = 45,#2,#v0_arr.shape[0],
            random_evaluation   = False,
            ))
        
        out = system.calc_OBEs(
            # t_int           = 30e-6, #30.4e-6
            dt              = 1e-9,#0.4e-9,#'auto',#1e-9,
            method          = 'RK45',
            # rtol=1e-4,atol=1e-6,
            magn_remixing   = True,
            verbose         = True,
            steadystate     = True,
            rounded         = False,
            freq_clip_TH    = 500,#500#,'auto',
            )
        
# %%
# Loading results
# ---------------

    plt.figure()
    
    for label in ["BaF", "CaF"]:
        fname = f"cooling_forces_{label}"
        
        system = open_object(fname)

        if label == 'BaF':
            F       = system.results.vals['F'][:,2] 
        else:
            F       = system.results.vals['F'][:,1]

        lamb    = system.levels.wavelengths.iloc[0,0]*1e-9
        Gamma   = system.levels.calc_Gamma()[0]
        offset  = dict(BaF=-7.92, CaF=-3.85)[label] # in terms of Gamma
        det     = (system.lasers[3].omega - 2*pi * c/lamb) / Gamma + offset
        
        # plotting
        plt.axhline(0, color='k', alpha=0.6)
        plt.plot(det, F / 1e-21, label = label)
    
        
    plt.xlabel('Detuning $\Delta$ ($\Gamma$)')
    plt.ylabel('Force ($10^{-21}$ N)')
    plt.legend()

# %%
# .. image:: /_figures/core/cooling_forces_fig1.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/cooling_forces_fig1.svg'