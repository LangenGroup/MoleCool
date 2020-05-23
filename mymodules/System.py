# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:17:56 2020

@author: Felix

This module contains the main class ``System`` which provides all information
about the Lasers, Levels and can carry out simulation calculations, e.g.
via the rate equations

Example
-------
Example for setting up a level and laser system with 3 repumping lasers and
calculating the dynamics::
    
    system = System(description='Test-Lasersystem')
    
    #pol     = 'lin'
    #pol     = 'sigmap'
    pol     = ('sigmap','sigmam')
    system.lasers.freq_pol_switch = 5e6
    
    system.lasers.add_sidebands(859.830e-9,20e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(895.699e-9,14e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(897.961e-9,14e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(900.238e-9,14e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    # system.lasers.add(859.830e-9,20e-3,pol)
    # system.lasers.add(895.699e-9,14e-3,pol)
    # system.lasers.add(897.961e-9,20e-3,pol)
    # system.lasers.add(900.238e-9,20e-3,pol)
    
    system.levels.grstates.add_grstate(nu=0,N=1)
    system.levels.grstates.add_grstate(nu=1,N=1)
    system.levels.grstates.add_grstate(nu=2,N=1)
    system.levels.grstates.add_grstate(nu=3,N=1)
    
    system.levels.grstates.add_lossstate(nu=4)
    
    system.levels.exstates.add_exstate(nu=0,N=0,J=.5,p=+1)
    system.levels.exstates.add_exstate(nu=1,N=0,J=.5,p=+1)
    system.levels.exstates.add_exstate(nu=2,N=0,J=.5,p=+1)
    
    nodetuned_list = [(0,0),(1,1),(2,2),(3,3)] 
    # system.N0 = np.array([*np.ones(system.levels.lNum),*np.zeros(system.levels.uNum)])
    system.calc_rateeqs(t_int=20e-6,dt=0.02e-6,perfect_resonance=False,nodetuned_list=nodetuned_list,velocity_dep=False,magn_remixing=False,calculated_by='Lucas')

"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c,h,hbar,pi,g
from BaFconstants import *
from Lasersystem import *
from Levelsystem import *
from math import floor
import _pickle as pickle

import matplotlib.pyplot as plt
# plt.rcParams['errorbar.capsize']=3
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('figure', figsize=(6.0,3.4))
# plt.rc('savefig', pad_inches=0.02, bbox='tight')
# plt.rc('axes', grid = True)
# plt.rc('grid', linestyle ='dashed', linewidth=0.5, alpha=0.7)
# plt.rcParams['axes.formatter.use_locale'] = True

h = 6.62607e-34#;%2*pi*1.054e-34;
c = 2.998e8
np.set_printoptions(precision=4,suppress=True)

#%%
class System:
    """System containing instances of ``Lasersystem`` and ``Levelsystem`` class.
    
    An instance of this class is the starting point to define a System on which
    one desires to calculate the time evolution via the rate equations.

    Example
    -------
    After the Lasers and Levels have been defined in the ``System`` object
    `system`, the rate equations can be applied and the results can be plotted
    via:
        
        >>> system.calc_rateeqs()
        >>> system.plot_N()
    """
    #scattphotons = 0
    #init_pop = 

    def __init__(self, description="Magneto-optical trapping forces for atoms and molecules with complex level structures"):
        self.lasers = Lasersystem()
        self.levels = Levelsystem()
        #self.particles = Particlesystem()
        self.description = description
        self.N0 = None #: initial population of all levels
        print("System is created as: {}".format(self.description))
        
    def calc_rateeqs(self,t_int=20e-6,dt=0.02e-6,
                     perfect_resonance=False, nodetuned_list=[],
                     velocity_dep=False,magn_remixing=False, calculated_by='Lucas'):
        """
        Calculates the occupations

        Parameters
        ----------
        t_int : float, optional
            interaction time in which the molecule is exposed to the lasers.
            The default is 20e-6.
        dt : float, optional
            time step of the output data. The default is 0.02e-6.
        perfect_resonance : bool, optional
            Property if a certain vibrational levels are in perfect resonance
            (with no detuning) with a specific lasers
            (defined in the `nodetuned_list`). The default is False.
        nodetuned_list : list(tuples), optional
            list of tuples (nu,p) for the specific vibrational groundstate level
            with `nu` and the laser with index `p`. The default is [].
        velocity_dep : TYPE, optional
            DESCRIPTION. The default is False.
        magn_remixing : bool, optional
            if True, the adjacent ground hyperfine levels are perfectly mixed
            by a magnetic field. The default is False.
        calculated_by : str, optional
            The branching ratios are different depending on the group or person 
            who has calculated these values. Either 'Lucas' or 'YanGroup'.
            The default is 'Lucas'.
        
        Note
        ----
        function creates attributes 
        
        ``self.N`` : solution of the time dependent populations N,
        ``self.Nscatt`` : time dependent scattering number,
        ``self.Nscattrate`` : time dependent scattering rate,
        ``self.photons``: totally scattered photons
        
        Returns
        -------
        None.

        """        
        # kp = np.array([laser.k for laser in lasers])
        # kp = np.tensordot(np.ones(3),np.tensordot(np.ones(3),kp,axes=0),axes=0)
        # if velocity_dep == False:
        #     R = R_func(v=0)
        # else: R = R_func
        lNum,uNum,pNum = self.levels.lNum, self.levels.uNum, self.lasers.pNum
        Gamma       = self.levels.exstates.Gamma
        self.sp     = np.array([ la.I / ( pi*c*h*Gamma/(3*la.lamb**3) ) 
                                for la in self.lasers ])
        
        self.t_eval = np.arange(0,t_int,dt)
        self.r      = np.zeros((lNum,uNum))
        self.rx1    = np.zeros((lNum,uNum,pNum))
        self.rx2    = np.zeros((lNum,uNum,pNum))
        self.R1     = np.zeros((lNum,uNum,pNum))
        self.R2     = np.zeros((lNum,uNum,pNum))
        self.delta  = np.zeros((lNum,uNum,pNum))
        if (self.N0 == None) and (lNum >= 12):
            self.N0     = np.zeros(lNum+uNum)
            self.N0[:12] = np.ones(12)/(12) #No initial population in lossstate
        
        for l in range(lNum):
            for u in range(uNum):
                gr,ex       = self.levels.grstates[l],self.levels.exstates[u]
                self.r[l,u] = branratios(gr, ex, calculated_by) * vibrbranch(gr, ex)
                
                for p in range(pNum):
                    self.delta[l,u,p]   = self.levels.freq_lu(l,u) - self.lasers[p].omega
                    #if nodetuning and abs(self.delta[l,u,p]) < 20e6*2*pi:
                    #    self.delta[l,u,p] = 0
                    if perfect_resonance:
                        # nodetuned_list contains tuples (nu,p) determining the
                        # groundstates nu being in perfect resonance with the lasers p
                        for nu, p_ in nodetuned_list:
                            if gr.nu == nu and p == p_:
                                self.delta[l,u,p] = 0
                    self.rx1[l,u,p]     = self.r[l,u] * selrule(gr,ex,self.lasers[p].pol1)
                    self.rx2[l,u,p]     = self.r[l,u] * selrule(gr,ex,self.lasers[p].pol2)
                    self.R1[l,u,p]      = Gamma/2 * (self.rx1[l,u,p]*self.sp[p]) / (1+ 4*(self.delta[l,u,p]/Gamma)**2)
                    self.R2[l,u,p]      = Gamma/2 * (self.rx2[l,u,p]*self.sp[p]) / (1+ 4*(self.delta[l,u,p]/Gamma)**2)
        
        if magn_remixing: self.M = magn_remix(self.levels.grstates)
        else: self.M = None
        
        tswitch = 1/self.lasers.freq_pol_switch
        
        #sum R1 & R2 over pNum:
        self.R1sum, self.R2sum = np.sum(self.R1,axis=2), np.sum(self.R2,axis=2)

        # solve initial value problem of the ordinary first order differential equation with scipy
        print('Solving ode...')
        sol = solve_ivp(ode, (0,t_int), self.N0, method='RK45', t_eval=self.t_eval,
                  dense_output=False, events=None, vectorized=False,
                  args=(lNum,uNum,pNum,Gamma,self.r,self.R1sum,self.R2sum,tswitch,self.M))
        
        self.N = sol.y #: solution of the time dependent populations N
        #: time dependent scattering number
        self.Nscatt = np.zeros(len(self.t_eval)-1)
        #: time dependent scattering rate
        self.Nscattrate = Gamma*np.sum(self.N[lNum:,:], axis=0) /(2*pi)
        for i in range(len(self.t_eval)-1):
            self.Nscatt[i] = np.sum( np.diff(self.t_eval[:i+2]) * self.Nscattrate[:i+1] )
        #: totally scattered photons
        self.photons = self.Nscatt[-1]
        print("Scattered Photons:",self.photons)
        
    def plot_all(self):
        self.plot_N()
        self.plot_Nscatt()
        self.plot_Nscattrate()
        self.plot_Nsum()
    def plot_N(self):
        plt.figure('N: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Populations $N$')
        for l in range(self.levels.lNum - 1):
            plt.plot(self.t_eval*1e6, self.N[l,:], 'b-')
        for u in range(self.levels.uNum):
            plt.plot(self.t_eval*1e6, self.N[system.levels.lNum+u,:], 'r-')
        plt.plot(self.t_eval*1e6, self.N[system.levels.lNum-1,:], 'y-')
        
    def plot_Nscatt(self):
        plt.figure('Nscatt: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Totally scattered photons')
        plt.plot(self.t_eval[:-1]*1e6, self.Nscatt, '-')
    
    def plot_Nscattrate(self):
        plt.figure('Nscattrate: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Photon scattering rate $\gamma\prime$ in MHz')
        plt.plot(self.t_eval*1e6, self.Nscattrate*1e-6, '-')
        
    def plot_Nsum(self):
        plt.figure('Nsum: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Population sum $\sum N_i$')
        plt.plot(self.t_eval*1e6, np.sum(self.N,axis=0), '-')

#%%
def save_object(obj,filename):
    """Save an entire class with all its attributes instead of saving a pyplot figure.
    
    Parameters
    ----------
    obj : object
        The object you want to save.
    filename : TYPE
        the filename to save the data.

    Returns
    -------
    None.
    """
    with open(filename,'wb') as output:
        pickle.dump(obj,output,-1)
def open_object(filename):
    with open(filename,'rb') as input:
        output = pickle.load(input)
    return output

#%%
def magn_remix(grs):
    """returns a matrix to perfectly remix all adjacent ground hyperfine levels
    by a magnetic field. The default is False.

    Parameters
    ----------
    grs : Groundstates object

    Returns
    -------
    array
        magnetic remixing matrix.

    """
    matr = np.zeros((grs.lNum,grs.lNum))
    for i in range(grs.lNum):
        for j in range(grs.lNum):
            if grs[i].name == 'Loss state' or grs[j].name == 'Loss state':
                if grs[i].name == grs[j].name:
                    matr[i,j] = 1
            elif grs[i].F == grs[j].F and grs[i].J == grs[j].J and abs(grs[i].mF-grs[j].mF) <= 1:
                matr[i,j] = 1
    return 1e9*matr

#%%
def selrule(gr,ex,pol):
    # from a loss state there's no dipole transition (stimulated by a laser)
    if gr.name == 'Loss state': return 0
    # only if the parity of both the ground and excited state are well defined
    # the parity-change and rotational momentum selection rules should apply:
    if not ( (gr.p + ex.p == 0) and (abs(gr.N - ex.N) <= 1) ):
        return 0
    elif pol == 'lin':
        if gr.mF == ex.mF: return 1
        else: return 0
    elif pol == 'sigmap':
        if gr.mF == ex.mF-1: return 1
        else: return 0
    elif pol == 'sigmam':
        if gr.mF == ex.mF+1: return 1
        else: return 0
        
#%%

def ode(t,N,lNum,uNum,pNum,Gamma,r,R1,R2,tswitch,M):
    dNdt = np.zeros(lNum+uNum)
    if floor(t/tswitch)%2 == 1: R=R1
    else: R=R2
    
    for l in range(lNum):
        dNdt[l] = Gamma*np.dot( r[l,:] , N[lNum:] )
        if np.any(M) != None: 
            for k in range(lNum):
                dNdt[l] -= M[l,k]*(N[l]-N[k])
        dNdt[l] += np.dot( R[l,:] , N[lNum:] - N[l] )
    for u in range(uNum):
        dNdt[lNum+u]  = -Gamma*N[lNum+u]
        dNdt[lNum+u] += np.dot( R[:,u] , N[:lNum]-N[lNum+u] )
                          
    return dNdt


#%%
#%%
if __name__ == '__main__':
    system = System(description='Test-Lasersystem')
    
    #pol     = 'lin'
    #pol     = 'sigmap'
    pol     = ('sigmap','sigmam')
    system.lasers.freq_pol_switch = 5e6
    
    system.lasers.add_sidebands(859.830e-9,20e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(895.699e-9,14e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(897.961e-9,14e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(900.238e-9,14e-3,pol,AOM_shift=-20.65e6, EOM_freq=39.33e6)
    # system.lasers.add(859.830e-9,20e-3,pol)
    # system.lasers.add(895.699e-9,14e-3,pol)
    # system.lasers.add(897.961e-9,20e-3,pol)
    # system.lasers.add(900.238e-9,20e-3,pol)
    
    system.levels.grstates.add_grstate(nu=0,N=1)
    system.levels.grstates.add_grstate(nu=1,N=1)
    system.levels.grstates.add_grstate(nu=2,N=1)
    system.levels.grstates.add_grstate(nu=3,N=1)
    
    system.levels.grstates.add_lossstate(nu=4)
    
    system.levels.exstates.add_exstate(nu=0,N=0,J=.5,p=+1)
    system.levels.exstates.add_exstate(nu=1,N=0,J=.5,p=+1)
    system.levels.exstates.add_exstate(nu=2,N=0,J=.5,p=+1)
    
    nodetuned_list = [(0,0),(1,1),(2,2),(3,3)] 
    # system.N0 = np.array([*np.ones(system.levels.lNum),*np.zeros(system.levels.uNum)])
    system.calc_rateeqs(t_int=20e-6,dt=0.02e-6,perfect_resonance=False,nodetuned_list=nodetuned_list,velocity_dep=False,magn_remixing=False,calculated_by='Lucas')



'''
print("Excitedstate ordering of columns:")
for u in range(4):
    ex = system.levels.exstates[u]
    print("F'={}, mF={}".format(ex.F,ex.mF))
for l in range(12):
    ratios = []
    for u in range(4):
        gr ,ex = system.levels.grstates[l],system.levels.exstates[u]
        ratios.append(branratios(gr,ex))
    print("J={}, F={}, mF={:+}: {}".format(gr.J,gr.F,gr.mF,ratios))
    
print('Selection rules'')
for l in range(13):
    list_=[]
    for u in range(4):
        gr,ex=system.levels.grstates[l],system.levels.exstates[u]
        list_.append(selrule(gr,ex,'sigmam'))
    print(list_)
'''
        