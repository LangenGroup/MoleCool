# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:27:42 2020

@author: fkogel

v1.4

Module with the BaF specific data to be able to perform molecule specific calculations.
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g,u

#%%
def get_mass():
    """returns the mass of the :math:`^{138}Ba^{19}F` molecule
    """
    return (138+19)*u
#%%
def get_Gamma():
    """returns the linewidth :math:`\Gamma` of the excited state
    :math:`A^2\Pi` of BaF as angular frequency.
    """
    return 2*pi*2.8421e6

#%%
def freq(gr,ex):
    """Gives the angular frequency difference between an ground state <gr> and 
    an excited state <ex> of BaF.

    Parameters
    ----------
    gr : :class:`~Levelsystem.Groundstate`
        ground state object
    ex : :class:`~Levelsystem.Excitedstate`
        excited state object
    Returns
    -------
    float
    
    Note
    ----
    if **<gr>** is a loss state, the returned value is arbitrarily set as 1.
    """
    # wavelengths of vibrational levels: cols: A(v') states, rows:X(v'') states
    lambda_vibr = np.array([
        [859.830, 828.903, 800.360],
        [895.699, 862.188, 831.350],
        [934.370, 897.961, 864.561],
        [976.181, 936.510, 900.238]])*1e-9
    # Energy level splitting of groundstates with quantum numbers J and F
    # Energies of excited levels are not resolved, and thus assumed to be equal
    J_F_dict = { 0.5 : {1: -94.9467, 0: -67.1338}, 1.5: {1: 22.7147, 2: 56.7660}}
    if gr.name == 'Loss state':
        return  1# arbitraryly specified frequency
    else:
        #return 2*pi*( c/lambda_dict[ex.nu][gr.nu] + 1e6*J_F_dict[gr.J][gr.F] )
        return 2*pi*(c/lambda_vibr[gr.nu,ex.nu] + 1e6*J_F_dict[gr.J][gr.F] )
    
#%%
def vibrbranch(gr,ex):
    """Returns the vibrational branching ratio between a groundstate <gr> and an
    excited state <ex> of BaF.

    Parameters
    ----------
    gr : Groundstate object.
    ex : Groundstate object.

    Returns
    -------
    float
    
    Note
    ----
    if the groundstate is a loss state, the returned branching ratio is the sum
    of all remaining vibrational levels which are not occupied by a ground state.
    """
    #first key=nu', second key=nu, value=Franck-Condon-Factor
    # see Summary_FCFs.pdf --> Lucas values
    branch_dict = {
                0:{0: 0.96748,
                   1: 0.03164,
                   2: 0.00087,
                   3: 9.76e-6,
                   4: 1.2e-7},
                1:{0: 0.04107,
                   1: 0.89385,
                   2: 0.06238,
                   3: 0.00266,
                   4: 0.00004},
                2:{0: 0.00050,
                   1: 0.08245,
                   2: 0.81959,
                   3: 0.09194,
                   4: 0.00540}
                }
    if gr.name == 'Loss state':
        return 1-np.sum([ branch_dict[ex.nu][nu] for nu in range(gr.nu)])
    else:
        return branch_dict[ex.nu][gr.nu]

            
#%%
def branratios(gr,ex,calculated_by='YanGroupnew'):
    """
    Returns the rotational branching ratios between specific hyperfine levels
    of a groundstate <gr> and an excited state <ex> of BaF.

    Parameters
    ----------
    gr : Groundstate object
    
    ex : Excitedstate object
    
    calculated_by : str, optional
        The branching ratios are different depending on the group or person 
        who has calculated these values. Either 'Lucas', 'YanGroup' or
        'YanGroupnew'. The default is 'YanGroupnew'.

    Returns
    -------
    float
    """
    # branching ratios for the transition nu'=0,N'=0,J'=.5,p=+ --> nu=0,N=1
    if gr.name == 'Loss state':
        return 1.0
    else:
        if gr.J == 0.5:
            if gr.F == gr.mF == 0:
                row = 0
            elif gr.F == 1:
                row = 2+gr.mF
        elif gr.J == 1.5:
            if gr.F == 1:
                row = 5+gr.mF
            elif gr.F == 2:
                row = 9+gr.mF
        if ex.F == ex.mF == 0:
            col = 0
        elif ex.F == 1:
            col = 2+ex.mF
        r = np.zeros((12,4))
        if calculated_by == 'Lucas':
            r[0,:] = [0,      0.2222, 0.2222, 0.2222] #J=1/2, F=0, MF=0
            
            r[1,:] = [0.1282, 0.2493, 0.2493, 0.0000] #J=1/2, F=1, MF=-1
            r[2,:] = [0.1282, 0.2493, 0.0000, 0.2493] #J=1/2, F=1, MF=-0
            r[3,:] = [0.1282, 0.0000, 0.2493, 0.2493] #J=1/2, F=1, MF=+1
        
            r[4,:] = [0.2051, 0.0007, 0.0007, 0.0000] #J=3/2, F=1, MF=-1
            r[5,:] = [0.2051, 0.0007, 0.0000, 0.0007] #J=3/2, F=1, MF=0
            r[6,:] = [0.2051, 0.0000, 0.0007, 0.0007] #J=3/2, F=1, MF=+1
            
            r[7,:] =  [0.0000, 0.1667, 0.0000, 0.0000] #J=3/2, F=2, MF=-2
            r[8,:] =  [0.0000, 0.0833, 0.0833, 0.0000] #J=3/2, F=2, MF=-1
            r[9,:] = [0.0000, 0.0278, 0.1111, 0.0278] #J=3/2, F=2, MF=0
            r[10,:] = [0.0000, 0.0000, 0.0833, 0.0833] #J=3/2, F=2, MF=+1
            r[11,:] = [0.0000, 0.0000, 0.0000, 0.1667] #J=3/2, F=2, MF=+2
            
        elif calculated_by == 'YanGroupnew':
            r[0,:] = [0,         2/9,    2/9,    2/9] #J=1/2, F=0, MF=0
            
            r[1,:] = [0.1282, 0.2493, 0.2493, 0.0000] #J=1/2, F=1, MF=-1
            r[2,:] = [0.1282, 0.2493, 0.0000, 0.2493] #J=1/2, F=1, MF=-0
            r[3,:] = [0.1282, 0.0000, 0.2493, 0.2493] #J=1/2, F=1, MF=+1
            
            r[4,:] = [0.20513333333333, 0.0007, 0.0007, 0.0000] #J=3/2, F=1, MF=-1
            r[5,:] = [0.20513333333333, 0.0007, 0.0000, 0.0007] #J=3/2, F=1, MF=0
            r[6,:] = [0.20513333333333, 0.0000, 0.0007, 0.0007] #J=3/2, F=1, MF=+1
            
            r[7,:] =  [0,   1/6,    0,      0   ] #J=3/2, F=2, MF=-2
            r[8,:] =  [0,   1/12,   1/12,   0   ] #J=3/2, F=2, MF=-1
            r[9,:] =  [0,   1/36,   1/9,    1/36] #J=3/2, F=2, MF=0
            r[10,:] = [0,   0,      1/12,   1/12] #J=3/2, F=2, MF=+1
            r[11,:] = [0,   0,      0,      1/6 ] #J=3/2, F=2, MF=+2
        
        elif calculated_by == 'YanGroup':
            r[0,:] = [0, 2/9, 2/9, 2/9] #J=1/2, F=0, MF=0
            
            r[1,:] = [0.2985, 0.1641, 0.1641, 0] #J=1/2, F=1, MF=-1
            r[2,:] = [0.2985, 0.1641, 0, 0.1641] #J=1/2, F=1, MF=-0
            r[3,:] = [0.2985, 0, 0.1641, 0.1641] #J=1/2, F=1, MF=+1
            
            r[4,:] = [0.0348, 0.0859, 0.0859, 0] #J=3/2, F=1, MF=-1
            r[5,:] = [0.0348, 0.0859, 0, 0.0859] #J=3/2, F=1, MF=0
            r[6,:] = [0.0348+9.5e-5+4.75e-6, 0, 0.0859, 0.0859] #J=3/2, F=1, MF=+1
            #r[6,:] = [0.0348, 0, 0.0859, 0.0859] #J=3/2, F=1, MF=+1
            
            r[7,:] =  [0, 1/6,  0,    0] #J=3/2, F=2, MF=-2
            r[8,:] =  [0, 1/12, 1/12, 0] #J=3/2, F=2, MF=-1
            r[9,:] = [0, 1/36, 1/9,  1/36] #J=3/2, F=2, MF=0
            r[10,:] = [0, 0,    1/12, 1/12] #J=3/2, F=2, MF=+1
            r[11,:] = [0, 0,    0,    1/6] #J=3/2, F=2, MF=+2
            
        return r[int(row),int(col)]

#%%
def FCF(nu_gr,nu_ex):
    """returns Franck-Condon-Factors
    (not needed for the calculation of the rate equations)

    Parameters
    ----------
    nu_gr : int
        vibrational state of the groundstate.
    nu_ex : int
        vibrational state of the excited state.

    Returns
    -------
    float
        FCF.
    
    Note
    ----
    values from Lucas' Masters thesis
    """
    #first key=nu', second key=nu, value=Franck-Condon-Factor
    #??? but these are other values than those in Summary_FCFs.pdf
    FCF_dict = {
                0:{0: 0.9508,
                   1: 0.0476,
                   2: 1.5e-3,
                   3: 2.7e-5,
                   4: 4.6e-7},
                1:{0: 0.0483,
                   1: 0.8539,
                   2: 0.0925,
                   3: 5.1e-3,
                   4: 1.3e-4},
                2:{0: 9.1e-4,
                   1: 0.0956,
                   2: 0.7581,
                   3: 0.1347,
                   4: 0.0104}
                }
    return FCF_dict[nu_ex][nu_gr]