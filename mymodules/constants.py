# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:27:42 2020

@author: fkogel

v2.5.3

Module containing specific contants of certain molecules, atoms or more general
systems. Theses constants will be imported within the classes
:py:class:`~Lasersystem.Lasersystem` and :class:`~Levelsystem.Levelsystem` to
be used in following calculations.

This module is constructed in such way so that it can be modified or extended
by other users to hold the constants for their specific level system.
Here, there is a function for every important quantity which all take a string
``name`` as input value. So e.g. if one want to get the electric dipole matrix
belonging to `BaF` the function call::
    
    dMat('BaF')
    
returns the matrix values with the respective row and column labels.
But for nicely displaying these constants and matrices please use the respective
functions in the class :class:`~Levelsystem.Levelsystem`, e.g. for the electric
dipole matrix :func:`~Levelsystem.Levelsystem.get_dMat`.
So, to nicely print all properties and constants for e.g. 138BaF try::
    
    levels = Levelsystem(load_constants='BaF')
    levels.add_all_levels(0)
    levels.print_properties()
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g,u
import _pickle as pickle
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) #directory where this script is stored.
# Using this directory path, the module System (and the others) can be imported
# from an arbitrary directory provided that the respective path is in the PYTHONPATH variable.
#%%
def dMat(name):
    """returns the electric dipole matrix of the electric dipole transition operator.
    """
    #ATTENTION: all label pairs (J,F,mF) or (J,F) have to be unique!
    if name == 'BaF':
        # decay ex -> gr: dipole_matrix_new1, gr -> ex: dipole_matrix_new2
        with open(os.path.join(script_dir,'dipole_matrix_new1.pkl'),'rb') as input:
            array = pickle.load(input).to_numpy() #2D-array of the dipole matrix
        row_labels = [[0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], #-> J
                      [0  ,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2], #-> F
                      [0  ,  -1,   0,  +1,  -1,   0,  +1,  -2,  -1,   0,   1,   2]] #-> mF
        column_labels = [[0.5,0.5,0.5,0.5], #-> J'
                         [0  ,1  ,1  ,1  ], #-> F'
                         [0  ,-1 ,0  ,+1]] #-> mF'
        
    elif name == 'CaF':#X-B transition
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
    """returns the reduced electric dipole matrix of the electric dipole transition operator.
    """
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
    elif name == '137BaF':
        array =[[-0.9074, -0.1325, -0.2635, -0.2712],
                [ 1.4981, -0.3807, -0.5068, -0.    ],
                [ 0.1358, -0.6456, -0.2039,  0.1946],
                [ 0.1968, -1.1891,  0.2889, -0.2963],
                [-0.0131,  0.0625, -0.4385, -0.3416],
                [-0.1789,  0.8507, -0.3824,  0.5273],
                [-0.3642,  0.5028, -0.2123,  1.1851],
                [ 1.1343, -0.2194, -0.9552,  0.0403],
                [-0.5021, -0.0945,  1.4665,  0.    ],
                [-0.0758,  0.9844,  0.3443,  0.    ],
                [-1.1424,  0.    ,  0.    , -0.    ]]
        row_labels = [[11, 12, 11, 12, 10,   21, 22, 21, 22, 23, 23 ],  #-> J = 10*G+F1
                      [1.5,2.5,0.5,1.5,0.5,  0.5,1.5,1.5,2.5,2.5,3.5]]  #-> F
        column_labels = [[2,    2,      1,      1], #-> J = F1
                         [2.5,  1.5,    1.5,    0.5]] #-> F
    else:
        return None
    return array, row_labels, column_labels

def vibrbranch(name):
    """returns the vibrational branching ratios for transitions between
    vibrational ground and excited state transitions of a certain electrical transition.
    """
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
    """Function for wavelengths and hyperfine frequencies of the transitions
    between all levels.
    """
    if name == 'BaF':
        # wavelengths of vibrational levels: cols: A(v') states, rows:X(v'') states
        lambda_vibr = [[859.830, 828.903, 800.360],
                       [895.699, 862.188, 831.350],
                       [934.370, 897.961, 864.561],
                       [976.181, 936.510, 900.238],
                       [1021.520,978.165, 938.663]]
        # Energy level splitting of groundstates with quantum numbers J and F
        # Energies of excited levels are not resolved, and thus assumed to be equal
        hyperfine_gr_labels = [[0.5,      0.5,     1.5,     1.5],   #-> J
                               [1,        0,       1,       2  ]]   #-> F
        hyperfine_gr        = [-94.9467, -67.1338, 22.7147, 56.7660] #->values in MHz (non-angular frequencies)
        
        hyperfine_ex_labels = [[0.5,     0.5],   #-> J'
                               [0,       1  ]]   #-> F'
        hyperfine_ex        =  [0.0,     0.0]    #-> values in MHz (non-angular frequencies)
        
        # rotational_gr       = [0.0, 12947.890, 38843.552, 77686.706] doesn't work yet!!!
    elif name == '137BaF':
        lambda_vibr = [[859.845]]
        hyperfine_gr_labels = [[11, 12, 11, 12, 10,   21, 22, 21, 22, 23, 23 ],  #-> J = 10*G+F1
                               [1.5,2.5,0.5,1.5,0.5,  0.5,1.5,1.5,2.5,2.5,3.5]]  #-> F
        hyperfine_gr = [10025.2498, 10029.9165, 10042.8014, 10054.3382, 10164.1612,
                        14635.2988, 14643.5158, 14677.1062, 14681.1442, 14729.4801,
                        14764.7988]
        hyperfine_ex_labels = [[2,    2,      1,      1],   #-> J = F1
                               [2.5,  1.5,    1.5,    0.5]] #-> F
        hyperfine_ex = [  0.    ,   0.    , 304.4994, 304.4994]
    elif name == 'CaF':
        # wavelengths of vibrational levels: cols: A(v') states, rows:X(v'') states
        lambda_vibr = [[531.0]]
        # Energy level splitting of groundstates with quantum numbers J and F
        # Energies of excited levels are not resolved, and thus assumed to be equal
        hyperfine_gr_labels = [[0.5,      0.5,     1.5,     1.5],   #-> J
                               [1,        0,       1,       2  ]]   #-> F
        hyperfine_gr        = [-97.6,   -22.9,  +23.9,    +48.8] #->values in MHz (non-angular frequencies)
        
        hyperfine_ex_labels = [[0.5,     0.5],   #-> J'
                               [0,       1  ]]   #-> F'
        hyperfine_ex        =  [0.0,     0.0]    #-> values in MHz (non-angular frequencies)
    else:
        return None
    return lambda_vibr, (hyperfine_gr,hyperfine_gr_labels), (hyperfine_ex,hyperfine_ex_labels)
        
def gfac(name):
    """Function for the magnetic g-factors of the ground and excited states.
    """
    if name == 'BaF':
        #mixed g factors of BaF (see Erratum: Structure, branching.. of Chen, Tao; Bu, Wenhao; Yan, Bo)
        # first key: J, second key: F of ground states and excited states
        row_labels_gr = [[0.5,  0.5,  1.5,  1.5],   #-> J
                         [  0,    1,    1,    2]]   #-> F
        array_gr      = [0.00, -0.51, 1.01, 0.50]
        
        row_labels_ex = [[0.5,  0.5],               #-> J'
                         [  0,    1]]               #-> F'
        array_ex      = [0.00, -0.20272] #parity-dep Zeeman shift #in H_Z: [0.,-0.186]
    elif name == '137BaF':
        row_labels_gr = [[11, 12, 11, 12, 10,   21, 22, 21, 22, 23, 23 ],  #-> J = 10*G+F1
                         [1.5,2.5,0.5,1.5,0.5,  0.5,1.5,1.5,2.5,2.5,3.5]]  #-> F
        array_gr      = [-0.251, -0.205, -0.439, -0.241,  0.063,
                          1.043,  0.447,  0.578,  0.325,  0.395,  0.286]
        
        row_labels_ex = [[2,    2,      1,      1],   #-> J = F1
                         [2.5,  1.5,    1.5,    0.5]] #-> F
        array_ex      = [-0.050,  -0.172, 0.033, 0.198] #[-0.00012, -0.00019,  0.00012,  0.00025]    
    elif name == 'CaF':
        row_labels_gr = [[0.5,  0.5,  1.5,  1.5],   #-> J
                         [  0,    1,    1,    2]]   #-> F
        array_gr      = [0.00, 0.64, -0.14, 0.5]
        
        row_labels_ex = [[0.5,  0.5],               #-> J'
                         [  0,    1]]               #-> F'
        array_ex      = [0.00, 1.0]        
    else:
        return None
    return (array_gr, row_labels_gr), (array_ex, row_labels_ex)

def Gamma(name):
    """Function for the natural decay rate or linewidth :math:`\Gamma` of the
    excited state (:math:`A^2\Pi` for BaF) as angular frequency.
    """
    if name == 'BaF':
        Gamma = 2*pi*2.8421e6
    elif name == '137BaF':
        Gamma = 2*pi*2.8421e6
    elif name == 'CaF': #X-B transition
        Gamma = 2*pi*6.34e6
    else:
        return 2*pi*1e6 #or is None better??
    return Gamma

def mass(name):
    """returns the mass (e.g. for the :math:`^{138}Ba^{19}F` molecule).
    """
    if name == 'BaF':
        mass = (138+19)*u
    elif name == '137BaF':
        mass = (138+19)*u
    elif name == 'CaF':
        mass = (40+19)*u
    else:
        return None
    return mass

def branratios(name,calculated_by='YanGroupnew'):
    """Returns the vibrational branching ratio between the ground and excited states.
    """
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
    