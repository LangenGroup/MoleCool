# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:27:42 2020

@author: fkogel

v2.0.0

Module with the BaF specific data to be able to perform molecule specific calculations.
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g,u
import _pickle as pickle
#%%
def dMat(name):
    if name == 'BaF':
        # decay ex -> gr: dipole_matrix_new1, gr -> ex: dipole_matrix_new2
        with open('dipole_matrix_new1'+'.pkl','rb') as input:
            array = pickle.load(input).to_numpy() #2D-array of the dipole matrix
        row_labels = [[0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], #-> J
                      [0  ,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2], #-> F
                      [0  ,  -1,   0,  +1,  -1,   0,  +1,  -2,  -1,   0,   1,   2]] #-> mF
        column_labels = [[0.5,0.5,0.5,0.5], #-> J'
                         [0  ,1  ,1  ,1  ], #-> F'
                         [0  ,-1 ,0  ,+1]] #-> mF'
        
    elif name == 'CaF':
        array = [[0         ,-0.5774    ,0          ,0          ],
                 [0         ,0.4082     ,-0.4082    ,0          ],
                 [0         ,-0.2357    ,0.4714     ,-0.2357    ],
                 [0         ,0          ,-0.4082    ,0.4082     ],
                 [0         ,0          ,0          ,-0.5774    ],
                 [-0.5743   ,-0.0421    ,-0.0421    ,0          ],
                 [0.5743    ,0.0421     ,0          ,-0.0421    ],
                 [-0.5743   ,0          ,0.0421     ,0.0421     ],
                 [0.0595    ,-0.4061    ,-0.4061    ,0          ],
                 [-0.0595   ,0.4061     ,0          ,-0.4061    ],
                 [0.0595    ,0          ,0.4061     ,0.4061     ],
                 [0         ,0.3333     ,0.3333     ,0.3333     ]]
        row_labels = [[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5], #-> J
                      [  2,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   0], #-> F
                      [ -2,  -1,   0,   1,   2,  -1,   0,  +1,  -1,   0,  +1,   0]] #-> mF
        column_labels = [[0.5,0.5,0.5,0.5], #-> J'
                         [0, 1, 1, 1], #-> F'
                         [0,-1, 0,+1]] #-> mF'
    else:
        return None
    return array, row_labels, column_labels

def dMat_red(name):
    if name == 'Lambda-system':
        array = [[1],
                 [1]]
        row_labels = [[0.5 ,1.5],   #-> J
                      [1   ,1]]     #-> F
        column_labels = [[0.5],     #-> J'
                         [1.0]]     #-> F'
        
    elif name == '2-level-system':
        array       = [[1]]
        row_labels  = [[0.5],       #-> J
                       [1  ]]       #-> F
        column_labels = [[0.5],     #-> J'
                         [0.0]]     #-> F'
    else:
        return None
    return array, row_labels, column_labels

def vibrbranch(name):
    # index of column:  vibrational quantum number nu' of the excited state
    # index of row:     vibrational quantum number nu  of the ground state
    # --> Each column should add to 1 approximately
    # see Summary_FCFs.pdf --> Lucas values
    if name == 'BaF':
        array = [[0.96748,      0.04107,    0.00050],
                 [0.03164,      0.89385,    0.08245],
                 [0.00087,      0.06238,    0.81959],
                 [9.76e-6,      0.00266,    0.09194],
                 [1.2e-7,       0.00004,    0.00540]]
    else:
        return None
    return array
    
def freq(name):
    if name == 'BaF':
        # wavelengths of vibrational levels: cols: A(v') states, rows:X(v'') states
        lambda_vibr = [[859.830, 828.903, 800.360],
                       [895.699, 862.188, 831.350],
                       [934.370, 897.961, 864.561],
                       [976.181, 936.510, 900.238]]
        # Energy level splitting of groundstates with quantum numbers J and F
        # Energies of excited levels are not resolved, and thus assumed to be equal
        hyperfine_gr_labels = [[0.5,      0.5,     1.5,     1.5],   #-> J
                               [1,        0,       1,       2  ]]   #-> F
        hyperfine_gr        = [-94.9467, -67.1338, 22.7147, 56.7660]
        
        hyperfine_ex_labels = [[0.5,     0.5],   #-> J'
                               [0,       1  ]]   #-> F'
        hyperfine_ex        =  [0.0,     0.0]
        
        # rotational_gr       = [0.0, 12947.890, 38843.552, 77686.706]
    else:
        return None
    return lambda_vibr, (hyperfine_gr,hyperfine_gr_labels), (hyperfine_ex,hyperfine_ex_labels)
        
def gfac(name):
    if name == 'BaF':
        #mixed g factors of BaF (see Erratum: Structure, branching.. of Chen, Tao; Bu, Wenhao; Yan, Bo)
        # first key: J, second key: F of ground states and excited states
        row_labels_gr = [[0.5,  0.5,  1.5,  1.5],   #-> J
                         [  0,    1,    1,    2]]   #-> F
        array_gr      = [0.00, -0.51, 1.01, 0.50]
        
        row_labels_ex = [[0.5,  0.5],               #-> J'
                         [  0,    1]]               #-> F'
        array_ex      = [0.00, 0.00]                #???????????????????
    else:
        return None
    return (array_gr, row_labels_gr), (array_ex, row_labels_ex)

def Gamma(name):
    if name == 'BaF':
        Gamma = 2*pi*2.8421e6
    elif name == 'CaF':
        Gamma = 2*pi*6.3e6
    else:
        return 2*pi*1e6 #or is None better??
    return Gamma

def mass(name):
    if name == 'BaF':
        mass = (138+19)*u
    elif name == 'CaF':
        mass = (40+19)*u
    else:
        return None
    return mass

def branratios(name,calculated_by='YanGroupnew'):
    if name == 'BaF':
        row_labels = [[0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], #-> J
                      [0  ,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2], #-> F
                      [0  ,  -1,   0,  +1,  -1,   0,  +1,  -2,  -1,   0,   1,   2]] #-> mF
        column_labels = [[0.5,0.5,0.5,0.5], #-> J'
                         [0  ,1  ,1  ,1  ], #-> F'
                         [0  ,-1 ,0  ,+1]] #-> mF'
        if calculated_by == 'Lucas':
            array = [
            [0,      0.2222, 0.2222, 0.2222], #J=1/2, F=0, MF=0
            [0.1282, 0.2493, 0.2493, 0.0000], #J=1/2, F=1, MF=-1
            [0.1282, 0.2493, 0.0000, 0.2493], #J=1/2, F=1, MF=-0
            [0.1282, 0.0000, 0.2493, 0.2493], #J=1/2, F=1, MF=+1
            [0.2051, 0.0007, 0.0007, 0.0000], #J=3/2, F=1, MF=-1
            [0.2051, 0.0007, 0.0000, 0.0007], #J=3/2, F=1, MF=0
            [0.2051, 0.0000, 0.0007, 0.0007], #J=3/2, F=1, MF=+1
            [0.0000, 0.1667, 0.0000, 0.0000], #J=3/2, F=2, MF=-2
            [0.0000, 0.0833, 0.0833, 0.0000], #J=3/2, F=2, MF=-1
            [0.0000, 0.0278, 0.1111, 0.0278], #J=3/2, F=2, MF=0
            [0.0000, 0.0000, 0.0833, 0.0833], #J=3/2, F=2, MF=+1
            [0.0000, 0.0000, 0.0000, 0.1667]] #J=3/2, F=2, MF=+2
            
        elif calculated_by == 'YanGroupnew':
            array = [
            [0,         2/9,    2/9,    2/9], #J=1/2, F=0, MF=0
            [0.1282, 0.2493, 0.2493, 0.0000], #J=1/2, F=1, MF=-1
            [0.1282, 0.2493, 0.0000, 0.2493], #J=1/2, F=1, MF=-0
            [0.1282, 0.0000, 0.2493, 0.2493], #J=1/2, F=1, MF=+1
            [0.20513333333333, 0.0007, 0.0007, 0.0000], #J=3/2, F=1, MF=-1
            [0.20513333333333, 0.0007, 0.0000, 0.0007], #J=3/2, F=1, MF=0
            [0.20513333333333, 0.0000, 0.0007, 0.0007], #J=3/2, F=1, MF=+1
            [0,   1/6,    0,      0   ], #J=3/2, F=2, MF=-2
            [0,   1/12,   1/12,   0   ], #J=3/2, F=2, MF=-1
            [0,   1/36,   1/9,    1/36], #J=3/2, F=2, MF=0
            [0,   0,      1/12,   1/12], #J=3/2, F=2, MF=+1
            [0,   0,      0,      1/6 ]] #J=3/2, F=2, MF=+2
    else:
        return None
    return array, row_labels, column_labels
    