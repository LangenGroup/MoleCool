# -*- coding: utf-8 -*-
"""
Optical cycling 12+4
====================

Transferring the insights in optical cycling from the simplest 3+1 level system
(see also :ref:`sphx_glr_auto_examples_core_optcycl_3+1levels.py`).
to the simplest level system of a molecule using :math:`^{138}\\text{BaF}` in this example.
"""
from MoleCool import System, np, open_object, plt
import os.path

# %%
# Initialize levels from BaF
# --------------------------
#
# We load the predefined constants of :math:`^{138}\text{BaF}` which are stored
# in the repository's folder `constants` in the file ``138BaF.json``.
# For the electronic ground and excited states, the states with vibrational
# quantum number ``v=0`` are loaded from the constants file.

if __name__ == '__main__':    
    if not os.path.isfile("optcycl_12+4levels.pkl"):
        
        system = System(load_constants='138BaF')
        
        system.levels.add_electronicstate('X','gs')
        system.levels.X.load_states(v=[0])
        system.levels.add_electronicstate('A','exs')
        system.levels.A.load_states(v=[0])
        
# %%
# Lasers and magnetic field
# -------------------------
        
        # intensity array
        I_arr       = np.logspace(np.log10(10000),np.log10(3.16),8)
        # Bfield array
        B_arr       = np.linspace(0,10,100+1)
        
        system.lasers.add_sidebands(
            lamb        = 859.83e-9,
            I           = I_arr,
            mod_freq    = 1e6,
            sidebands   = [94.9, 66.9, -39.5, -56.5],
            ratios      = [28,   15,    17,    40],
            )
        
        system.Bfield.turnon(
            strength    = B_arr*1e-4,
            angle       = 60,
            )
# %%
# Setting up OBEs
# ---------------
#
# We can tune some of the steady state and multiprocessing parameters for this
# rather long calculation. 
# At the end the results should be saved into a .pkl file.

        system.steadystate['t_ini']     = 15e-6
        system.steadystate['period']    = None
        system.steadystate['maxiters']  = 50
        system.steadystate['condition'] = [0.1,5]
        
        system.multiprocessing['maxtasksperchild']  = 20
        system.multiprocessing['show_progressbar']  = True
        system.multiprocessing['savetofile']        = True
        
        system.calc_OBEs(t_int=15e-6,dt=1e-9,method='RK45',rtol=1e-4,atol=1e-6,
                         magn_remixing=True,verbose=True,steadystate=True,
                         rounded=False,freq_clip_TH=30)
    
# %%
# Loading data and plotting
# -------------------------
# 
# At the end the whole system instance from the ``.pkl`` file is imported and
# the OBE data stored in ``sys_load.results.vals`` is plotted.

    sys_load    = open_object("optcycl_12+4levels")
    
    B_arr       = sys_load.Bfield.strength*1e4
    I_arr       = sys_load.lasers.I_sum
    
    plt.figure('(12 + 4) system')
    for i,I in enumerate(I_arr):
        plt.plot(B_arr, 2*sys_load.results.vals['Ne'][:,i],
                  label=r'${:.0f}$'.format(I), color=plt.cm.plasma(i/(len(I_arr)-1)))
    
    plt.xlabel(r'Magnetic field strength (G)')
    plt.ylabel('Scattering rate $R_{sc}$ ($\Gamma/2$)')
    plt.legend(title='$I_\mathrm{tot}$ (W/m$^2$)',
               bbox_to_anchor=(1,1), loc='upper left')
    
    print('Mean execution time per Bfield, Intensity, and core:',
          '{:.2f} s'.format(sys_load.results.vals['exectime'].mean()))

# %%
# .. image:: /_figures/core/optcycl_12+4levels_fig1.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/optcycl_12+4levels_fig1.svg'