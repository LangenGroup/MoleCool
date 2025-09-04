"""
Optical cycling 3+1
===================

text here
"""

# %%
from MoleCool import System, np, plt, pi, hbar
from MoleCool.spectra import Molecule

# required for multiprocessing: function definition for returning desired quantities
def return_fun(system):
    return {'Ne' : system.N[-1,-1], 'exectime' : system.exectime}

mol = Molecule()
# %%
if __name__ == '__main__':
    # create empty system instance
    system=System(description='Simple3+1_AppC2')
    
    # construct level system
    system.levels.add_electronicstate('X','gs')
    system.levels.add_electronicstate('A','exs',Gamma=1.0)
    system.levels.X.add(F=1)
    system.levels.A.add(F=0)
    system.levels.X.gfac.iloc[0] = 1.0 # set ground state g factor to 1.0
    system.levels.print_properties() # print all properties of the levels
# %%
# constants and arrays to iterate

    Gamma   = system.levels.calc_Gamma()[0] # default natural linewidth
    Om_L    = hbar*Gamma/system.Bfield.mu_B # Bfield inducing Larmor frequency Gamma
    B_arr   = np.linspace(0.,2.,20*3+1)**2 *Om_L # Bfield array
    Om_arr  = np.logspace(np.log10(7),np.log10(0.25),8) *Gamma # Rabi frequencies
    
    # set up laser system and magnetic field
    system.lasers.add(lamb=860e-9,pol='lin',freq_shift=+1/4*Gamma/2/pi,freq_Rabi=Om_arr)
    system.Bfield.turnon(strength=B_arr,direction=[0,1,1])
    
    # simulate dynamics with OBEs using multiprocessing
    system.calc_OBEs(t_int=50e-6,method='DOP853', magn_remixing=True,verbose=False,
                     steadystate=True,mp=True,return_fun=return_fun)
    
# %%
# plotting

    plt.figure('(3 + 1) system')
    for j,Om in enumerate(Om_arr):
        # the excited state populations from the OBEs are saved in system.results
        plt.plot(B_arr/Om_L, 2*system.results[0]['Ne'][:,j],
                 label=r'${:.2f}$'.format(Om/Gamma), color=plt.cm.plasma(j/(len(Om_arr)-1)))
    plt.legend(title='$\Omega$ [$\Gamma$]')
    plt.xlabel(r'Larmor frequency $\omega_L$ [$\Gamma$]')
    plt.ylabel('Scattering rate $R_{sc}$ [$\Gamma/2$]')
    print('Mean execution time per Bfield, Rabi frequency, and core:',
          '{:.2f} s'.format(system.results[0]['exectime'].mean()))
    
    # plt.savefig("Fig4_3+1system_Bfield")
# %%
# .. image:: /_figures/core/optcycl_3+1levels_fig1.svg
#    :alt: Demo plot
#    :align: center

# sphinx_gallery_thumbnail_path = '_figures/core/optcycl_3+1levels_fig1.svg'