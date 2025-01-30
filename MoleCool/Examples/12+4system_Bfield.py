# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:02:22 2022

@author: fkogel

tested with v3.2.2
"""

from System import *

def return_fun1(obj):
    if obj.calcmethod=='OBEs':
        NeNum = obj.levels.uNum
        return {'Ne'        : obj.N[-NeNum:,:].sum(axis=0).mean(),
                'F'         : obj.F[2,:].mean(axis=-1),#/(hbar*2*pi/860e-9*2*pi*2.8421e6/2),
                'Nscattrate': obj.Nscattrate[0,:].mean()}
    elif obj.calcmethod=='rateeqs':
        return {'Ne'        : obj.N[-4:,-(obj.t.size//10):].sum(axis=0).mean(),#-period:].sum(axis=0).mean(),
                'F'         : obj.F[2,-(obj.t.size//10):].mean(),#,  -period:].mean(),
                'exectime'  : obj.exectime}
#%%  
if __name__ == '__main__':
    import os.path
    if not os.path.isfile("12+4system_Bfield.pkl"):
        system = System(load_constants='138BaF')
        
        # intensity array
        I_arr       = np.logspace(np.log10(10000),np.log10(3.16),8)
        # Bfield array
        B_arr       = np.linspace(0,10,100+1)
        
        t_int       = 15e-6
        system.steadystate['t_ini'] = 15e-6
        system.steadystate['period'] = None#'standingwave' #wavelength in System was changed!
        system.steadystate['maxiters'] = 50
        system.steadystate['condition'] = [0.1,5]
        mp          = True
        system.multiprocessing['maxtasksperchild'] = 20
        system.multiprocessing['show_progressbar'] = True
        system.multiprocessing['savetofile'] = True
        # system.multiprocessing['processes'] = v0_arr.shape[0]
    
        #%% initialize system with levels
        system.levels.add_electronicstate('X','gs')
        system.levels.X.load_states(v=[0])
        system.levels.add_electronicstate('A','exs')
        system.levels.A.load_states(v=[0])
        
        #%% lasers & magnetic field
        system.lasers.add_sidebands(lamb=859.83e-9, mod_freq=1e6, I=I_arr,
                                    sidebands = [94.9, 66.9, -39.5, -56.5],
                                    ratios    = [28,   15,    17,    40])
        
        system.Bfield.turnon(strength=B_arr*1e-4, angle=60)
        
        #%% OBEs
        system.calc_OBEs(t_int=t_int,dt=1e-9,method='RK45',rtol=1e-4,atol=1e-6,
                         magn_remixing=True,verbose=True,steadystate=True,
                         mp=mp,rounded=False,freq_clip_TH=30,return_fun=return_fun1)
    
    #%% plotting
    sys_load = open_object("12+4system_Bfield")
    plt.figure('(12 + 4) system')
    B_arr = sys_load.Bfield.strength*1e4
    I_arr = sys_load.lasers.I_sum
    for i,I in enumerate(I_arr):
        # the excited state populations from the OBEs are saved in system.results
        plt.plot(B_arr, 2*sys_load.results[0]['Ne'][:,i],
                  label=r'${:.0f}$'.format(I), color=plt.cm.plasma(i/(len(I_arr)-1)))
    plt.legend(title='$I_\mathrm{tot}$ [W/m$^2$]', bbox_to_anchor=(1,1), loc='upper left')
    plt.xlabel(r'Magnetic field strength [G]')
    plt.ylabel('Scattering rate $R_{sc}$ [$\Gamma/2$]')
    print('Mean execution time per Bfield, Intensity, and core:',
          '{:.2f} s'.format(sys_load.results[0]['exectime'].mean()))
    
    plt.savefig("Fig5_12+4system_Bfield")

