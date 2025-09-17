# -*- coding: utf-8 -*-
"""
Optical pumping in 171YbOH
--------------------------

Beyond diatomic molecules, recent years have seen major advances in the direct
laser cooling and control of polyatomic species.
A particularly relevant case is ytterbium monohydroxide (YbOH), whose
odd isotopologues :math:`^{173}\\text{YbOH}` and :math:`^{171}\\text{YbOH}` 
are of great interest as sensitive probes for nuclear properties and
tests of fundamental symmetries.
Motivated by these applications, optical cycling in these isotopologues has
recently been demonstrated for the first time.

In this tutorial, we will model the optical cycling process in 171YbOH
to highlight the key mechanisms that enable photon scattering in such
polyatomic systems.
"""
# %%
# Import and function definition
# ------------------------------
from MoleCool import System, np, plt, save_object, open_object
from MoleCool.spectra import multiindex_filter as filt
from MoleCool.tools import gaussian
import os.path

def fun(system):
    t_cut  = (system.lasers[-1].r_k[0]-3e-3)/system.v0[0]
    Nscatt = np.where(system.t>t_cut, system.Nscattrate[0], 0)
    Nscatt = np.interp(system.t_plt, system.t, Nscatt, right=0, left=0)
    return {'Nscatt':Nscatt}

# %%
# Level system initialization
# ---------------------------
if __name__ == '__main__':
    if not os.path.isfile("YbOH.pkl"):
        system = System(load_constants='171YbOH')
        
        system.levels.add_electronicstate('X','gs')
        system.levels.add_electronicstate('A','exs')
        
        system.levels.X.load_states(v=[0,1])
        system.levels.A.load_states(v=[0])
        
        system.levels.X.add_lossstate()
        
# %%
# Since we added a loss state which includes a vibrational state with the
# vibrational quantum number which is by 1 greater than the maximum ``v``.
# So, we deliberately must add another vibrational level in the pandas.DataFrame
# ``system.levels.vibrbranch`` and ``system.levels.wavelengths`` for the loading
# to be working.

        system.levels.vibrbranch.loc[('X',2),:]  = 1 - system.levels.vibrbranch.sum()
        system.levels.wavelengths.loc[('X',2),:] = 700
        
        # system.levels.print_properties()
        
# %%
# Transition spectrum
# -------------------
# Let's have a look how the transition spectrum looks like:
        
        system.levels.plot_transition_spectrum(
            wavelengths = [float(system.levels.wavelengths.loc[('X',0)])*1e-9],
            subplot_sep = 2000,
            N_points    = 1000,
            relative_to_wavelengths = True,
            # QuNrs       = [['G','F'],['F']], # formatting for legend
            legend      = False,
            )

# %%
# .. image:: /_figures/core/YbOH_fig1.svg
#    :alt: Demo plot
#    :align: center

# %%
# Initial population and velocity
# -------------------------------
        
        system.levels.X.set_init_pops({'v=0':1/1.11,'v=1':0.11/1.11})
        # system.levels.X.set_init_pops({'v=0,G=1,F1=1':1/1.11*6/24,
        #                                 'v=1,G=1,F1=1':0.11/1.11*6/24})
        
        system.set_v0(
            # v0 = [170,0,0],
            v0 = np.linspace(80,400,100),
            direction = 'x',
            )
        
        system.t_plt = np.linspace(0,0.2e-3,800)
        
# %%
# Calculation routine
# -------------------
        colors  = dict(ro='C0', rc='C1', nopump='k')
        D_exp   = dict(ro=0.48+0.17*np.array([+1,-1]), rc=7.4+1.3*np.array([+1,-1]))
        labels  = dict(rc='rc pumping',ro='ro pumping',nopump='no pumping')
        x_pump  = 5e-3
        x_probe = 17e-3
        P_pump  = 10e-3        

        
        for P, F1_pump, label in zip([1e-8,P_pump,P_pump], [0,0,1], ['nopump','rc','ro']):
            del system.lasers[:]
            #pump
            system.lasers.add(
                lamb    = system.levels.wavelengths.loc[('X',0),('A',0)]*1e-9,
                P       = P,
                FWHM    = 5e-3,
                k       = [0,0,1],
                r_k     = [x_pump,0,0],
                freq_shift = 1e6*(filt(system.levels.A.freq, dict(F1=F1_pump)).mean() - filt(system.levels.X.freq, dict(G=1,F1=1)).mean()),
                )
            
            #probe
            system.lasers.add(
                lamb    = system.levels.wavelengths.loc[('X',1),('A',0)]*1e-9,
                P       = 0.1e-3,
                w       = 1e-3,
                k       = [0,0,1],
                r_k     = [x_probe,0,0],
                freq_shift = 1e6*(filt(system.levels.A.freq, dict(F1=0)).mean() - filt(system.levels.X.freq, dict(G=1,F1=1)).mean()),
                )
        
            # system.lasers.plot_I_2D(ax='z',limits=([0,17e-3],[-7e-3,7e-3]))
            # 
            # system.lasers[-1].I
            # system.draw_levels(QuNrs_sep=['v'])
            
            #%%
            system.multiprocessing.update({
                'processes' : 2,
                # 'maxtasksperchild' : None,
                # 'show_progressbar' : True,
                'savetofile'       : False
                })
            
            system.calc_rateeqs(
                # t_int=10e-6,
                t_int           = x_probe/80,
                magn_remixing   = True,
                magn_strength   = 7,
                trajectory      = True,
                position_dep    = True,
                max_step        = 1e-6,
                verbose         = False,
                return_fun      = fun,
                )
            
            save_object(system, 'YbOH_'+label)
    
# %%
# Loading data and plotting
# -------------------------
    
    plt.figure()
    unit_x = 1e6
    lifs_arr = dict()
    for label in ['nopump','ro','rc']:
        system = open_object('YbOH_'+label)
        
        LIFs_i = system.results.vals['Nscatt']*gaussian(system.v0[:,0],a=1,x0=180,std=40)[:,None]*1e-3
        LIFs = LIFs_i.sum(axis=0)
        if label == 'rc':
            plt.plot(system.t_plt*unit_x, 2.2 * LIFs_i[::(LIFs_i.shape[0]//25),:].T,
                    c='grey', alpha=0.3)
        # for i, ls in enumerate(['-','--']):
        plt.plot(system.t_plt*unit_x, LIFs,
                 ls=dict(rc='-',ro='-',nopump='--')[label],
                 label=labels[label],
                 c=colors[label],
                 )
        lifs_arr[label] = LIFs
    
        if label in ['ro','rc']:
            lif_pump = lifs_arr[label].max()
            lif_nopump = lifs_arr['nopump'].max()
            D = (lif_pump - lif_nopump)/lif_nopump
            
            
            lif_pump_exp = (D_exp[label]+1)*lif_nopump
            
            plt.axhspan(*lif_pump_exp, alpha=0.25, color=colors[label])
            print(f'Max ({label})',lif_pump/lif_nopump)
            print(f'D  ({label})',D)
            
            
    plt.xlim(left=0.9*x_probe/system.v0.max()*unit_x)
    plt.xlabel('Time (us)')   
    plt.ylabel('Photon scattering rate (kHz)')
    plt.legend()
    
# %%
# .. image:: /_figures/core/YbOH_fig2.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/YbOH_fig2.svg'