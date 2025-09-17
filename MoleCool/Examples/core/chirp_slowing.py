# -*- coding: utf-8 -*-
"""
Chirped slowing
===============

Slowing of a molecular beam using radiation pressure.

- Rate equations
- 100 particles
- ...
"""
from MoleCool import System, np, open_object, save_object, plt, c
import os.path
from MoleCool.tools import gaussian
from copy import deepcopy

def return_fun(system):
    return dict(
        vx_end  = system.v[0,-1], 
        photons  = system.photons[0],
        )

# %%
# Initialize levels from BaF
# --------------------------
#
# We load the predefined constants of :math:`^{138}\text{BaF}` which are stored
# in the repository's folder `constants` in the file ``138BaF.json``.
# For the electronic ground and excited states, the states with vibrational
# quantum number ``v=0`` are loaded from the constants file.

if __name__ == '__main__':    
    if not os.path.isfile("chirp_slowing.pkl"):
    
        system      = System(load_constants='138BaF')
        system.levels.add_all_levels(v_max=3)
        del system.levels['B']
        
# %%
# Velocity distribution
# ---------------------
        
        system.set_v0(
            v0          = np.linspace(70, 310, 3000),
            direction   = 'x',
            )
# %%
# Laser wavelengths and chirp rates
# ---------------------------------

        beta_arr    = np.arange(0,3*4+1,3) * 1e6/1e-3 #MHz per ms
        lamb_arr    = np.array([system.levels.wavelengths.iloc[i,j]
                                for i,j in zip([0,1,2,3],[0,0,1,2])])*1e-9
        freq_arr    = c/lamb_arr
        freq_chirp  = -190/(c+190) * freq_arr
        lamb_start  = c/ ( freq_arr + freq_chirp)
        
        sidebands_kwargs = dict(
            sinusoidal  = dict(sidebands    = [-2,-1,1,2],
                               ratios       = [0.8,1,1,0.8],
                               mod_freq     = 38.4e6,
                               offset_freq  = 19e6,
                               ),
            perfect_fit = dict(sidebands    = [94.9, 66.9, -39.5, -56.5],
                               ratios       = [28,   15,    17,    40],
                               mod_freq     = 1e6,
                               ),
            )
# %%
# Calculation for two different types of sidebands
# ------------------------------------------------
        # system.multiprocessing['processes'] = 30  
        labels  = ['sinusoidal', 'perfect_fit']
        data    = dict()
        for label in labels:
            
            del system.lasers[:]
            
            for lamb in lamb_start:
                system.lasers.add_sidebands(
                    lamb    = lamb,
                    I       = 7000,
                    pol     = 'lin',
                    beta    = beta_arr*lamb_arr[0]/lamb,
                    k       = [-1,0,0],
                    r_k     = [0,0,0], 
                    **sidebands_kwargs[label],
                    )
            
            system.calc_rateeqs(
                t_int           = 9e-3,
                method          = 'LSODA',
                magn_remixing   = True,
                magn_strength   = 8,
                trajectory      = True,
                verbose         = False,
                return_fun      = return_fun,
                )
            
            data[label] = deepcopy(system)
        save_object(data, "chirp_slowing")
# %%
# Loading results
# ---------------

    data = open_object("chirp_slowing")
    
    plt.figure()
    
    for ls, label in zip([':', '-'], ['sinusoidal', 'perfect_fit']):
        
        system      = data[label]
        v0          = system.v0
        beta_arr    = system.lasers.getarr('beta')[0]
        vx_end      = system.results.vals['vx_end'].T
        photons     = system.results.vals['photons'].T
    
        mu          = np.array([190,0])
        sigma       = np.array([32.5448,  3.2545])
        fac         = np.exp(-(v0[:,0]-mu[0])**2/(2*sigma[0]**2))
        # fac         = gaussian(v0[:,0], x0=mu[0], std=sigma[0])
        bins        = 200
        y_offset    = 20
    
        for i,beta in enumerate(beta_arr):
            out = np.histogram(v0[:,0], bins=bins, weights=fac)
            # plt.plot(out[1][1:], out[0]+i*y_offset,
            #         color='grey',ls='-', alpha=0.4)
            plt.plot(out[1][1:], out[0]*0+i*y_offset,
                    color='k',ls='-', alpha=0.4)
            
            plt.fill_between(out[1][1:], i*y_offset, out[0]+i*y_offset,
                            color='grey',ls='-', alpha=0.1)
            
            out = np.histogram(vx_end[i,:], bins=bins, weights=fac)
            plt.plot(out[1][1:],out[0]+i*y_offset,
                    label='$\\beta$ = {:.0f}'.format(beta/(1e6/1e-3)) if ls=='-' else None,
                    color=f"C{i}", ls=ls)
            
    plt.xlim(81, 259)
    plt.ylim(bottom=-y_offset*0.1)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Number of molecules per bin')

    plt.legend(title='Chirping rate\n(MHz/ms)')

# %%
# .. image:: /_figures/core/chirp_slowing_fig1.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/chirp_slowing_fig1.svg'