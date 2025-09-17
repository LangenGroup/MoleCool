"""
Optical cycling on a Type-II transition 
=======================================

In this tutorial, we explore the simplest example of a type-II optical transition:
the :math:`F=1 \\rightarrow F'=0` level system.
Here, the ground state has three magnetic sublevels (:math:`m=−1,0,+1`),
while the excited state has only a single sublevel (:math:`m'=0`).

When the system is driven by light of a fixed polarization, only one of the
ground sublevels couples to the excited state.
Spontaneous decay, however, redistributes population across all three ground
sublevels, leading to population trapping in the two dark states that are not
coupled by the laser. As a result, the scattering process eventually halts.

To illustrate how this trapping can be overcome, we will include the effect of
a magnetic field tilted with respect to the laser polarization, which mixes
the ground sublevels and keeps the optical cycle active.
"""

# %%
# Import and function definition
# ------------------------------
from MoleCool import System, np, plt, pi, hbar

# required for multiprocessing: function definition for returning desired quantities
def return_fun(system):
    return {'Ne' : system.N[-1,-1], 'exectime' : system.exectime}

# %%
# Level system initialization
# ---------------------------
if __name__ == '__main__':
    # create empty system instance
    system = System(description='optcycl_simple3+1')
    
    # construct level system
    system.levels.add_electronicstate('X','gs')
    system.levels.add_electronicstate('A','exs',Gamma=1.0)
    system.levels.X.add(F=1)
    system.levels.A.add(F=0)
    system.levels.X.gfac.iloc[0] = 1.0 # set ground state g factor to 1.0
    system.levels.print_properties() # print all properties of the levels
# %%
# Constants and iteration arrays 
# ------------------------------

    Gamma   = system.levels.calc_Gamma()[0] # default natural linewidth
    Om_L    = hbar*Gamma/system.Bfield.mu_B # Bfield inducing Larmor frequency Gamma
    B_arr   = np.linspace(0.,2.,20*3+1)**2 *Om_L # Bfield array
    Om_arr  = np.logspace(np.log10(7),np.log10(0.25),8) *Gamma # Rabi frequencies
    
# %%
# Laser, Bfield and Evaluation
# ----------------------------

    # set up laser system and magnetic field
    system.lasers.add(pol='lin', freq_shift=+1/4*Gamma/2/pi, freq_Rabi=Om_arr)
    system.Bfield.turnon(strength=B_arr, direction=[0,1,1])
    
    # simulate dynamics with OBEs using multiprocessing
    system.calc_OBEs(t_int=50e-6,method='DOP853', magn_remixing=True, verbose=False,
                     steadystate=True, return_fun=return_fun)
    
# %%
# Plotting of results
# -------------------
# The results from the OBEs are stored as an :class:`~.tools.Results_OBEs_rateeqs`
# object in ``system.results``.
# For examples, the excited state populations are saved in the dictionary
# ``system.results.vals`` with the shape of the iterating variables (see
# ``system.results.iters``), i.e. first and second dimension are of the same
# shape as ``B_arr`` and ``Om_arr``, respectively.

    plt.figure('(3 + 1) system')
    for j,Om in enumerate(Om_arr):
        plt.plot(B_arr/Om_L, 2*system.results.vals['Ne'][:,j],
                 label=r'${:.2f}$'.format(Om/Gamma), color=plt.cm.plasma(j/(len(Om_arr)-1)))
    plt.legend(title='$\Omega$ ($\Gamma$)')
    plt.xlabel(r'Larmor frequency $\omega_L$ ($\Gamma$)')
    plt.ylabel('Scattering rate $R_{sc}$ ($\Gamma/2$)')
    
    print('Mean execution time per Bfield, Rabi frequency, and core:',
          '{:.2f} s'.format(system.results.vals['exectime'].mean()))
    
# %%
# .. image:: /_figures/core/optcycl_3+1levels_fig1.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/optcycl_3+1levels_fig1.svg'