# -*- coding: utf-8 -*-
"""
Created on Tue June 09 10:17:00 2020

@author: fkogel

v2.1.0

This module contains the main class :class:`System` which provides all information
about the Lasers, Levels and can carry out simulation calculations, e.g.
via the rate equations.

Examples
--------
Example for setting up a level and laser system with the cooling and 3 repumping
lasers and calculating the dynamics::
    
    system = System(description='Test-Lasersystem')
    
    #pol     = 'lin'
    #pol     = 'sigmap'
    pol     = ('sigmap','sigmam')
    system.lasers.freq_pol_switch = 5e6
    
    system.lasers.add_sidebands(859.830e-9,20e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(895.699e-9,14e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(897.961e-9,14e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6)
    system.lasers.add_sidebands(900.238e-9,14e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6)
    # use the following lasers without sidebands when `perfect_resonance`
    # input argument of the `calc_rateeqs` function is set to `True`.
    # system.lasers.add(859.830e-9,20e-3,pol)
    # system.lasers.add(895.699e-9,14e-3,pol)
    # system.lasers.add(897.961e-9,14e-3,pol)
    # system.lasers.add(900.238e-9,14e-3,pol)
    
    system.levels.grstates.add_grstate(nu=0,N=1)
    system.levels.grstates.add_grstate(nu=1,N=1)
    system.levels.grstates.add_grstate(nu=2,N=1)
    system.levels.grstates.add_grstate(nu=3,N=1)
    
    system.levels.grstates.add_lossstate(nu=4)
    
    system.levels.exstates.add_exstate(nu=0,N=0,J=.5,p=+1)
    system.levels.exstates.add_exstate(nu=1,N=0,J=.5,p=+1)
    system.levels.exstates.add_exstate(nu=2,N=0,J=.5,p=+1)
    
    # nodetuned_list contains tuples (nu_gr,nu_ex,p) determining the ground-/
    # excited state nu being in perfect resonance with the pth laser
    nodetuned_list = [(0,0,0),(1,0,1),(2,1,2),(3,2,3)]
    # system.N0 = np.array([*np.ones(system.levels.lNum),*np.zeros(system.levels.uNum)])
    system.calc_rateeqs(t_int=20e-6,perfect_resonance=False,nodetuned_list=nodetuned_list)
    
Example for a Molecule with a initial velocity and position of :math:`v_x=200m/s`
and :math:`r_x=-2mm` which is transversely passing two cooling lasers with
a repumper each in the distance :math:`4mm`::
    
    system = System(description='Test-Lasersystem')
    
    pol     = 'lin'
    system.lasers.freq_pol_switch = 5e6
    system.v0 = np.array([200,0,0])
    system.r0 = np.array([-2e-3,0,0])
    
    system.lasers.add_sidebands(859.830e-9,20e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6,k=[0,1,0])
    system.lasers.add_sidebands(895.699e-9,14e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6,k=[0,1,0])
    system.lasers.add_sidebands(859.830e-9,20e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6,k=[0,1,0],r_k=[4e-3,0,0])
    system.lasers.add_sidebands(895.699e-9,14e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6,k=[0,1,0],r_k=[4e-3,0,0])
        
    system.levels.grstates.add_grstate(nu=0,N=1)    
    system.levels.grstates.add_grstate(nu=1,N=1)
    
    system.levels.grstates.add_lossstate(nu=2)
    
    system.levels.exstates.add_exstate(nu=0,N=0,J=.5,p=+1)
    
    # nodetuned_list contains tuples (nu_gr,nu_ex,p) determining the ground-/
    # excited state nu being in perfect resonance with the pth laser
    nodetuned_list = [(0,0,0),(1,0,1),(2,1,2),(3,2,3)]
    system.calc_rateeqs(t_int=40e-6,perfect_resonance=False,
                        nodetuned_list=nodetuned_list,magn_remixing=False,
                        velocity_dep=True,position_dep=True)
    system.plot_Nscattrate()
    system.plot_Nscatt()
"""
import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
from scipy.constants import c,h,hbar,pi,g,u,physical_constants
from sympy.physics.wigner import clebsch_gordan,wigner_3j,wigner_6j
from Lasersystem import *
from Levelsystem import *
from math import floor
import _pickle as pickle
import time
from numba import jit, prange
import sys, os

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
# plt.rcParams['errorbar.capsize']=3
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('figure', figsize=(6.0,3.4))
# plt.rc('savefig', pad_inches=0.02, bbox='tight')
# plt.rc('axes', grid = True)
# plt.rc('grid', linestyle ='dashed', linewidth=0.5, alpha=0.7)
# plt.rcParams['axes.formatter.use_locale'] = True

np.set_printoptions(precision=4,suppress=True)

#%%
class System:
    """System containing instances of :py:class:`~Lasersystem.Lasersystem`
    and :class:`~Levelsystem.Levelsystem` class.
    
    An instance of this class is the starting point to define a System on which
    one desires to calculate the time evolution via the rate equations.

    Example
    -------
    After the Lasers and Levels have been defined in the :class:`System` object
    `system`, the rate equations can be applied and the results can be plotted
    via:
        
        >>> system.calc_rateeqs()
        >>> system.plot_N()
    """
    def __init__(self, description=None,load_constants='BaF'):
        """creates empty instances of the class :class:`~Lasersystem.Lasersystem`
        and the class :class:`~Levelsystem.Levelsystem` as ``self.lasers`` and
        ``self.levels``.

        Parameters
        ----------
        description : str, optional
            A short description of this System can be added. If not provided,
            the attribute is set to the name of the respective executed main
            python file.

        Returns
        -------
        None.

        """
        self.lasers = Lasersystem()
        self.levels = Levelsystem(load_constants=load_constants)
        #self.particles = Particlesystem()
        if description == None:
            self.description = os.path.basename(sys.argv[0])[:-3]
        else:
            self.description = description
        self.N0 = np.array([]) #: initial population of all levels
        self.v0 = np.array([]) #: initial velocity of the particle
        self.r0 = np.array([]) #: initial position of the particle
        print("System is created with description: {}".format(self.description))
        
    def calc_rateeqs(self,t_int=20e-6,t_start=0.,dt=None,t_eval = [],
                     perfect_resonance=False, nodetuned_list=[],
                     magn_remixing=False, magn_strength=8,
                     velocity_dep=False, position_dep=False,
                     calculated_by='YanGroupnew',verbose=True,**kwargs):
        """Calculates the time evolution of the single level occupations with
        rate equations.        

        Parameters
        ----------
        t_int : float, optional
            interaction time in which the molecule is exposed to the lasers.
            The default is 20e-6.
        t_start : float, optional
            starting time when the ode_solver starts the calculation. Useful
            for the situation when e.g. all cooling lasers are shut off at a 
            specific time t1, so that a new calculation with another laser
            configuration (e.g. including only a probe laser) can be started
            at t_start=t1 to continue the simulation. The default is 0.0.
        dt : float, optional
            time step of the output data. So in this case the ODE solver will
            decide at which time points to calculate the solution.
            The default is None.
        t_eval : list or numpy array, optional
            If it desired to get the solution of the ode solver only at 
            specific time points, the `t_eval` argument can be used to specify 
            these points. If `_eval` is given, the `dt` argument is ignored.
            The default is [].
        perfect_resonance : bool, optional
            Property if a certain vibrational levels are in perfect resonance
            (with no detuning) with a specific lasers
            (defined in the `nodetuned_list`). The default is False.
        nodetuned_list : list(tuples), optional
            list of tuples (nu_gr,nu_ex,p) for a specific vibrational ground 
            and excited state level with `nu_gr` and `nu_ex` and for the laser
            with index `p`. For such transition the detuning of laser `p` is
            set to zero. The default is [].
        magn_remixing : bool, optional
            if True, the adjacent ground hyperfine levels are perfectly mixed
            by a magnetic field. The default is False.
        magn_strength : float, optional
            measure of the magnetic field strength (i.e. the magnetic remixing
            matrix is multiplied by 10^magn_strength). Reasonable values are
            between 6 and 9. The default is 8.
        velocity_dep : bool, optional
            whether to calculate the velocity of a particle which changes due 
            to a momentum transfer and thus affects e.g. the scattering rate
            due to the Doppler shift.
            The default is False.
        position_dep : bool, optional
            whether to take the Gaussian intensity distribution of the laser
            beams into account. The default is False.
        calculated_by : str, optional
            The branching ratios are different depending on the group or person 
            who has calculated these values. Either 'Lucas', 'YanGroup' or
            'YanGroupnew'. The default is 'YanGroupnew'.
        verbose : bool, optional
            whether to print additional information like execution time or the
            scattered photon number. The default is True.
        **kwargs : keyword arguments, optional
            other options of the `solve_ivp` scipy function can be specified
            (see homepage of scipy for further information).
        
        Note
        ----
        function creates attributes 
        
        * ``self.N`` : solution of the time dependent populations N,
        * ``self.Nscatt`` : time dependent scattering number,
        * ``self.Nscattrate`` : time dependent scattering rate,
        * ``self.photons``: totally scattered photons
        * ``self.args``: input arguments of the call of this function
        * ``self.t`` : times at which the solution was calculated
        * ``self.v`` : calculated velocities of the molecule at times `self.t`
          (only given if velocity_dep == True)
        * ``self.r`` : calculated positions of the molecule at times `self.t`
          (only given if velocity_dep == True)
        
        Returns
        -------
        None.

        """
        #___input arguments of this called function
        self.args = locals()
        
        #___parameters belonging to the lasers
        lNum,uNum,pNum = self.levels.lNum, self.levels.uNum, self.lasers.pNum
        Gamma       = self.levels.exstates.Gamma
        self.sp     = np.array([ la.I / ( pi*c*h*Gamma/(3*la.lamb**3) ) 
                                for la in self.lasers ])
        self.k      = np.array([ la.k*la.kabs for la in self.lasers ])
        self.w      = np.array([la.w for la in self.lasers ])
        self.w_cylind = np.array([la.w_cylind for la in self.lasers ])
        self.r_k    = np.array([ np.array(la.r_k,dtype=float) for la in self.lasers ])
        self.beta   = np.array([la.beta for la in self.lasers ])
        #polarization switching time
        tswitch = 1/self.lasers.freq_pol_switch
        
        #___parameters belonging to the levels (and partially to the lasers)
        self.m      = self.levels.mass
        self.r_     = self.levels.calc_branratios(calculated_by)
        self.rx1    = np.zeros((lNum,uNum,pNum))
        self.rx2    = np.zeros((lNum,uNum,pNum))
        self.R1     = np.zeros((lNum,uNum,pNum))
        self.R2     = np.zeros((lNum,uNum,pNum))
        self.delta  = np.zeros((lNum,uNum,pNum))
        for l in range(lNum):
            for u in range(uNum):
                gr,ex       = self.levels.grstates[l],self.levels.exstates[u]
                for p in range(pNum):
                    la = self.lasers[p]
                    self.delta[l,u,p]   = -(self.levels.calc_freq()[l,u] - la.omega)
                    #if nodetuning and abs(self.delta[l,u,p]) < 20e6*2*pi:
                    #    self.delta[l,u,p] = 0
                    if perfect_resonance:
                        # nodetuned_list contains tuples (nu_gr,nu_ex,p)
                        # determining the ground/ excited states nu being in
                        # perfect resonance with the lasers p
                        for nu_gr, nu_ex, p_ in nodetuned_list:
                            if gr.nu == nu_gr and ex.nu == nu_ex and p == p_:
                                self.delta[l,u,p] = 0
                    self.rx1[l,u,p]     = self.r_[l,u] * selrule(gr,ex,la.pol1)
                    self.rx2[l,u,p]     = self.r_[l,u] * selrule(gr,ex,la.pol2)
                    self.R1[l,u,p]      = Gamma/2 * (self.rx1[l,u,p]*self.sp[p]) / (1+ 4*(self.delta[l,u,p])**2/Gamma**2)
                    self.R2[l,u,p]      = Gamma/2 * (self.rx2[l,u,p]*self.sp[p]) / (1+ 4*(self.delta[l,u,p])**2/Gamma**2)
        #sum R1 & R2 over pNum:
        self.R1sum, self.R2sum = np.sum(self.R1,axis=2), np.sum(self.R2,axis=2)
        
        #___magnetic remixing of the ground states. An empty array is left for no B-field 
        if magn_remixing: self.M = magn_remix(self.levels.grstates,magn_strength)
        else: self.M = np.array([[],[]])
        
        #___determine the time points at which the ODE solver should evaluate the equations
        if len(t_eval) != 0: self.t_eval = np.array(t_eval)
        else:   
            if dt != None and dt < t_int:
                self.t_eval = np.linspace(t_start,t_start+t_int,int(t_int/dt)+1)
            else:
                self.t_eval = None
        
        #___specify the initial (normalized) occupations of the levels
        self.N0 = np.array(self.N0, dtype=float)
        if len(self.N0) == 0:
            N0_indices = [i for i,st in enumerate(self.levels.grstates) if st.nu==0]    
            self.N0      = np.zeros(lNum+uNum)
            for i in N0_indices:
                self.N0[i] = 1.0
        self.N0 /= self.N0.sum()
        
        #___depenending on the position dependence two different ODE evaluation functions are called
        if velocity_dep or position_dep:
            self.y0      = np.array([*self.N0, *self.v0, *self.r0])
        else:
            self.y0      = self.N0
        
        # ---------------Ordinary Differential Equation solver----------------
        #solve initial value problem of the ordinary first order differential equation with scipy
        if verbose: print('Solving ode with rate equations...', end='')
        start_time = time.perf_counter()
        if not velocity_dep and not position_dep:
            sol = solve_ivp(ode0_jit, (t_start,t_start+t_int), self.y0,
                    t_eval=self.t_eval, **kwargs,
                    args=(lNum,uNum,pNum,Gamma,self.r_,self.R1sum,self.R2sum,
                          tswitch,self.M))
        else:
            sol = solve_ivp(ode1_jit, (t_start,t_start+t_int), self.y0,
                    t_eval=self.t_eval, max_step = 10e-6, **kwargs,
                    args=(lNum,uNum,pNum,Gamma,self.r_,self.rx1,self.rx2,
                          self.delta,self.sp,self.w,self.w_cylind,
                          self.k,self.r_k,self.m,tswitch,self.M,position_dep,self.beta))
            #velocity v and position r
            self.v = sol.y[-6:-3]
            self.r = sol.y[-3:]
        #execution time for the ODE solving
        self.exectime = time.perf_counter()-start_time
        self.calcmethod = 'rateeqs'
        if verbose: print(" execution time: {:.4f} seconds".format(self.exectime))
        
        #___compute several physical variables using the solution of the ODE
        #: array of the times at which the solutions are calculated
        self.t = sol.t
        #: solution of the time dependent populations N
        self.N = sol.y[:lNum+uNum]
        #: time dependent scattering rate
        self.Nscattrate = Gamma*np.sum(self.N[lNum:,:], axis=0)
        #: time dependent scattering number
        self.Nscatt = cumtrapz(self.Nscattrate, self.t, initial = 0.0)
        #: totally scattered photons
        self.photons = self.Nscatt[-1]
        if verbose:
            print("Scattered Photons:",self.photons)
            dev = abs(self.N[:,-1].sum() -1)
            if dev > 1e-8:
                print('WARNING: the sum of the occupations does not remain stable! Deviation: {:.2E}'.format(                    dev))
    #%%        
    def plot_all(self):
        self.plot_N(); self.plot_Nscatt(); self.plot_Nscattrate(); self.plot_Nsum()
    def plot_N(self,figname=None,figsize=(12,5),smallspacing=0.001):
        if figname == None:
            plt.figure('N ({}): {}, {}, {}'.format(
                self.calcmethod,self.description,self.levels.description,
                self.lasers.description), figsize=figsize)
        else: plt.figure(figname,figsize=figsize)

        lNum, uNum = self.levels.lNum, self.levels.uNum
        colors_l = pl.cm.jet(np.linspace(0.05,0.95,lNum))
        colors_u = pl.cm.jet(np.linspace(0,1,uNum))
        for i,grstate in enumerate(self.levels.grstates):
            if grstate.name == 'Loss state':
                label = 'Loss state'
                colors_l[i] = np.array([0,0,0,1])
            else:
                label='$g: J={:1.1f}, F={:1.1f}, mF={:+1.1f}$'.format(grstate.J,grstate.F,grstate.mF)
            plt.plot(self.t*1e6,self.N[i,:]+smallspacing*i, label=label,c=colors_l[i])
        for i,exstate in enumerate(self.levels.exstates):
            label =   '$e: J={:1.1f}, F={:1.1f}, mF={:+1.1f}$'.format(exstate.J,exstate.F,exstate.mF)
            plt.plot(self.t*1e6,self.N[lNum+i,:]+smallspacing*i,'-.',c=colors_u[i],label=label)
        
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Populations $N$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small',labelspacing=-0.0)
        plt.tight_layout()  
        
    def plot_Nscatt(self):
        plt.figure('Nscatt: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Totally scattered photons')
        plt.plot(self.t*1e6, self.Nscatt, '-')
    
    def plot_Nscattrate(self):
        plt.figure('Nscattrate: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Photon scattering rate $\gamma\prime$ in MHz')
        plt.plot(self.t*1e6, self.Nscattrate*1e-6, '-')
        
    def plot_Nsum(self):
        plt.figure('Nsum: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Population sum $\sum N_i$')
        plt.plot(self.t*1e6, np.sum(self.N,axis=0), '-')
    
    def plot_dt(self):
        if 'method' in self.args['kwargs']: method = self.args['kwargs']['method']
        else: method = 'RK45'
        plt.figure('dt: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('timestep d$t$ in s')
        plt.plot(self.t[:-1]*1e6,np.diff(self.t),label=method)
        plt.yscale('log')
        plt.legend()
        
    def plot_v(self):
        plt.figure('v: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('velocity $v_x, v_y, v_z$ in m/s')
        plt.plot(self.t*1e6, self.v[0,:], '-',label='$v_x$')
        plt.plot(self.t*1e6, self.v[1,:], '-',label='$v_y$')
        plt.plot(self.t*1e6, self.v[2,:], '-',label='$v_z$')
        plt.legend()
    
    def plot_r(self):
        plt.figure('r: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('position $r_x, r_y, r_z$ in m')
        plt.plot(self.t*1e6, self.r[0,:], '-',label='$r_x$')
        plt.plot(self.t*1e6, self.r[1,:], '-',label='$r_y$')
        plt.plot(self.t*1e6, self.r[2,:], '-',label='$r_z$')
        plt.legend()
        
    def plot_FFT(self,only_sum=True,start_time=0.0):
        FT_sum = 0
        t_int = self.args['t_int']
        # start_time = 0.5
        for i,st in enumerate(self.levels):
            FT = (np.fft.rfft(self.N[i,int(self.t.size*start_time):]).real)**2
            mean_zero = FT[int(FT.size/4):].mean()
            start = np.where(np.diff(FT)>0)[0][0]
            if start < 3: start = 3
            FT[:start] = mean_zero
            if not only_sum:
                FT[np.where(FT<2*mean_zero)[0]] = mean_zero
                plt.plot(np.arange(FT.size)/(t_int*(1-start_time))*1e-6,FT*1.**i)
            else:
                FT_sum += FT
        if only_sum:
            self.FT_sum = FT_sum
            plt.plot(np.arange(FT.size)/(t_int*(1-start_time))*1e-6,self.FT_sum)
        plt.yscale('log')
        plt.xlabel('Frequency $f$ in MHz')
        plt.ylabel('Power spectrum of the FFT')
    
    def calc_Rabi_freqs(self):
        #calculates detuning-weighted, averaged Rabi frequencies for every laser (with 2*pi included)
        Gamma = self.levels.exstates.Gamma
        Rabi_freqs = []
        for k,la in enumerate(self.lasers):
            Rabi_freqs_arr = np.abs(self.G[k]/np.sqrt(2)*np.dot(self.dMat, self.f[k,:])* Gamma)
            weights= np.exp(-4*(self.omega_eg-self.omega_k[k])**2/(Gamma**2))
            weights = weights*np.sign(Rabi_freqs_arr) * (np.dot(self.dMat, self.f[k,:]))**2
            # print(Rabi_freqs_arr,'\n',weights)
            Rabi_freqs.append( (Rabi_freqs_arr*weights).sum()/weights.sum() )
        self.Rabi_freqs = np.array(Rabi_freqs)
        return self.Rabi_freqs
    
    #%%
    def calc_OBEs(self,t_int=20e-6,t_start=0.,dt=None,t_eval = [],
                  perfect_resonance=False, nodetuned_list=[],
                  magn_remixing=False, magn_strength=8,
                  velocity_dep=False, position_dep=False,
                  calculated_by='Lucas',verbose=True,**kwargs):
        
        #___input arguments of this called function
        self.args = locals()
        
        #___parameters belonging to the lasers
        lNum,uNum,pNum = self.levels.lNum, self.levels.uNum, self.lasers.pNum
        Gamma       = self.levels.exstates.Gamma
        self.G      = np.array([ la.I / ( pi*c*h*Gamma/(3*la.lamb**3) ) 
                                for la in self.lasers ])**0.5
        self.f      = np.array([la.f_q for la in self.lasers])
        self.k      = np.array([ la.k*la.kabs for la in self.lasers ])
        self.phi    = np.array([la.phi for la in self.lasers])
        self.omega_k    = np.array([la.omega for la in self.lasers])
        #polarization switching time
        tswitch = 1/self.lasers.freq_pol_switch        

        #___parameters belonging to the levels (and partially to the lasers)
        self.m      = self.levels.mass 
        #electric dipole matrix
        self.dMat = self.levels.calc_dMat()
        #frequency differences between the ground and excited states
        self.omega_eg   = self.levels.calc_freq()
        
        #___magnetic remixing of the ground states and excited states
        if magn_remixing:
            betaB  = self.betaB
        else:
            betaB  = np.array([0.,0.,0.])
        #magnetic dipole matrix of the magnetic dipole operator mu
        self.muMat = self.levels.calc_muMat()
        #indices for the magnetic zeeman sublevels. Needed for magnetic remixing
        self.M_indices = self.levels.calc_M_indices()
                    
        if perfect_resonance:
            # nodetuned_list contains tuples (nu_gr,nu_ex,p)
            # determining the ground/ excited states nu being in
            # perfect resonance with the lasers p
            for l in range(lNum):
                for u in range(uNum):
                    gr,ex = self.levels.grstates[l],self.levels.exstates[u]
                    for p in range(pNum):
                        for nu_gr, nu_ex, p_ in nodetuned_list:
                            if gr.nu == nu_gr and ex.nu == nu_ex and p == p_:
                                self.omega_eg[l,u] = self.omega_k[p]
        
        #___determine the time points at which the ODE solver should evaluate the equations    
        if len(t_eval) != 0: self.t_eval = np.array(t_eval)
        else:   
            if dt != None and dt < t_int:
                self.t_eval = np.linspace(t_start,t_start+t_int,int(t_int/dt)+1)
            else:
                self.t_eval, T_eval = None, None
        if np.all(self.t_eval) != None: T_eval = self.t_eval * Gamma
        
        #___specify the initial (normalized) occupations of the levels
        N = lNum + uNum
        N0mat = np.zeros((N,N),dtype=np.complex64)
        self.N0 = np.array(self.N0, dtype=float)
        if len(self.N0) == 0:
            N0_indices = [i for i,st in enumerate(self.levels.grstates) if st.nu==0]    
            self.N0      = np.zeros(lNum+uNum)
            for i in N0_indices:
                self.N0[i] = 1.0
        self.N0 /= self.N0.sum()
        #transform these initial values into the density matrix elements
        N0mat[(np.arange(N),np.arange(N))] = self.N0   
        N0_vec = np.zeros( N*(N+1) )
        count = 0
        for i in range(N):
            for j in range(i,N):
                N0_vec[count]   = N0mat[i,j].real
                N0_vec[count+1] = N0mat[i,j].imag
                count += 2
                
        #___depenending on the position dependence two different ODE evaluation functions are called
        if velocity_dep or position_dep:
            self.y0      = N0_vec#np.array([*self.N0, *self.v0, *self.r0])
        else:
            self.y0      = N0_vec
        
        # ---------------Ordinary Differential Equation solver----------------
        # solve initial value problem of the ordinary first order differential equation with scipy
        if verbose: print('Solving ode with OBEs...', end='')
        start_time = time.perf_counter()
        if not velocity_dep and not position_dep:
            sol = solve_ivp(ode0_OBEs, (t_start*Gamma,(t_start+t_int)*Gamma),
                            self.y0, t_eval=T_eval, **kwargs,
                            args=(lNum,uNum,pNum,self.G,self.f,self.omega_eg/Gamma,
                                  self.omega_k/Gamma,betaB,self.dMat,self.muMat,
                                  self.M_indices))
        else:
            sol = solve_ivp(ode1_OBEs, (t_start*Gamma,(t_start+t_int)*Gamma),
                            self.y0, t_eval=T_eval, **kwargs,
                            args=(lNum,uNum,pNum,self.G,self.f,self.omega_eg/Gamma,
                                  self.omega_k/Gamma,betaB,self.dMat,self.muMat,
                                  self.M_indices,self.k/Gamma,self.v0,self.phi))
        #execution time for the ODE solving
        self.exectime = time.perf_counter()-start_time
        if verbose: print(" execution time: {:.4f} seconds".format(self.exectime))
        self.calcmethod = 'OBEs'
        
        #___compute several physical variables using the solution of the ODE
        #: array of the times at which the solutions are calculated
        self.t = sol.t/Gamma
        #: solution of the time dependent density matrix elements
        self.ymat    = np.zeros((N,N,self.t.size),dtype=np.complex64)
        y_vec = sol.y
        count   = 0
        for i in range(N):
            for j in range(i,N):
                self.ymat[i,j,:] = y_vec[count] + 1j* y_vec[count+1]
                count += 2     
        self.ymat    += np.conj(np.transpose(self.ymat,axes=(1,0,2))) #is diagonal remaining purely real or complex?
        self.ymat[(np.arange(N),np.arange(N))] *=0.5
        #: solution of the time dependent populations N
        self.N = np.real(self.ymat[(np.arange(N),np.arange(N))])
        #: time dependent scattering rate
        self.Nscattrate = Gamma*np.sum(self.N[lNum:,:], axis=0)
        #: time dependent scattering number
        self.Nscatt = cumtrapz(self.Nscattrate, self.t, initial = 0.0)
        #: totally scattered photons
        self.photons = self.Nscatt[-1]
        if verbose:
            print("Scattered Photons:",self.photons)
            dev = abs(self.N[:,-1].sum() -1)
            if dev > 1e-6:
                print('WARNING: the sum of the occupations does not remain stable! Deviation: {:.2E}'.format(dev))
    
    def add_magnfield(self,strength,direction=[0,0,1]):
        if strength >= 10e-4:
            print('WARNING: linear Zeeman shifts are only a good approx for B<10G.')
        ex,ey,ez = np.array(direction) / np.linalg.norm(direction)
        eps = np.array([+(ex - 1j*ey)/np.sqrt(2), ez, -(ex + 1j*ey)/np.sqrt(2)])
        self.betaB = eps*strength/ ( hbar*self.levels.exstates.Gamma/physical_constants['Bohr magneton'][0])

#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode0_OBEs(T,y_vec,lNum,uNum,pNum,G,f,om_eg,om_k,betaB,dMat,muMat,M_indices):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex64)
    
    ymat    = np.zeros((N,N),dtype=np.complex64)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(lNum):
        for b in range(uNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c in range(lNum):
                        dymat[a,lNum+b] += 1j*G[k]*f[k,q+1]/2/(2**0.5) *np.exp(1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[a,c]
                    for c_ in range(uNum):
                        dymat[a,lNum+b] -= 1j*G[k]*f[k,q+1]/2/(2**0.5) *np.exp(1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,lNum+b]
                for n in M_indices[1][b]:
                    dymat[a,lNum+b] += 1j*(-1.)**q* betaB[q+1]* muMat[1][b,n,-q+1]* ymat[a,lNum+n]
                for m in M_indices[0][a]:
                    dymat[a,lNum+b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c in range(lNum):
                        dymat[lNum+a,lNum+b] += 1j*G[k]/2/(2**0.5)*(
                            f[k,q+1] *np.exp(1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[lNum+a,c]
                            - np.conj(f[k,q+1]) *np.exp(-1j*T*(om_eg[c,a]-om_k[k]))* dMat[c,a,q+1]* ymat[c,lNum+b])
                for n in M_indices[1][b]:
                    dymat[lNum+a,lNum+b] += 1j*(-1.)**q* betaB[q+1]* muMat[1][b,n,-q+1]* ymat[lNum+a,lNum+n]
                for m in M_indices[1][a]:
                    dymat[lNum+a,lNum+b] -= 1j*(-1.)**q* betaB[q+1]* muMat[1][m,a,-q+1]* ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c_ in range(uNum):
                        dymat[a,b] -= 1j*G[k]/2/(2**0.5)*(
                            f[k,q+1] *np.exp(1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,b]
                            - np.conj(f[k,q+1]) *np.exp(-1j*T*(om_eg[b,c_]-om_k[k]))* dMat[b,c_,q+1]* ymat[a,lNum+c_])
                for n in M_indices[0][b]:
                    dymat[a,b] += 1j*(-1.)**q* betaB[q+1]* muMat[0][b,n,-q+1]* ymat[a,n]
                for m in M_indices[0][a]:
                    dymat[a,b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,b]
                for c_ in range(uNum):
                    for c__ in range(uNum):
                        dymat[a,b] += dMat[a,c_,q+1]* dMat[b,c__,q+1]* np.exp(1j*T*(om_eg[a,c_]-om_eg[b,c__]))* ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode1_OBEs(T,y_vec,lNum,uNum,pNum,G,f,om_eg,om_k,betaB,dMat,muMat,M_indices,kvec,v,phi):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex64)
    
    ymat    = np.zeros((N,N),dtype=np.complex64)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(lNum):
        for b in range(uNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c in range(lNum):
                        dymat[a,lNum+b] += 1j*G[k]*f[k,q+1]/2/(2**0.5) *np.exp(1j*(T*np.dot(kvec[k,:],v)+phi[k])) *np.exp(1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[a,c]
                    for c_ in range(uNum):
                        dymat[a,lNum+b] -= 1j*G[k]*f[k,q+1]/2/(2**0.5) *np.exp(1j*(T*np.dot(kvec[k,:],v)+phi[k])) *np.exp(1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,lNum+b]
                for n in M_indices[1][b]:
                    dymat[a,lNum+b] += 1j*(-1.)**q* betaB[q+1]* muMat[1][b,n,-q+1]* ymat[a,lNum+n]
                for m in M_indices[0][a]:
                    dymat[a,lNum+b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c in range(lNum):
                        dymat[lNum+a,lNum+b] += 1j*G[k]/2/(2**0.5)*(
                            f[k,q+1] *np.exp(1j*(T*np.dot(kvec[k,:],v)+phi[k])) *np.exp(1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[lNum+a,c]
                            - np.conj(np.exp(1j*(T*np.dot(kvec[k,:],v)+phi[k])) *f[k,q+1]) *np.exp(-1j*T*(om_eg[c,a]-om_k[k]))* dMat[c,a,q+1]* ymat[c,lNum+b])
                for n in M_indices[1][b]:
                    dymat[lNum+a,lNum+b] += 1j*(-1.)**q* betaB[q+1]* muMat[1][b,n,-q+1]* ymat[lNum+a,lNum+n]
                for m in M_indices[1][a]:
                    dymat[lNum+a,lNum+b] -= 1j*(-1.)**q* betaB[q+1]* muMat[1][m,a,-q+1]* ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c_ in range(uNum):
                        dymat[a,b] -= 1j*G[k]/2/(2**0.5)*(
                            f[k,q+1] *np.exp(1j*(T*np.dot(kvec[k,:],v)+phi[k])) *np.exp(1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,b]
                            - np.conj(np.exp(1j*(T*np.dot(kvec[k,:],v)+phi[k])) *f[k,q+1]) *np.exp(-1j*T*(om_eg[b,c_]-om_k[k]))* dMat[b,c_,q+1]* ymat[a,lNum+c_])
                for n in M_indices[0][b]:
                    dymat[a,b] += 1j*(-1.)**q* betaB[q+1]* muMat[0][b,n,-q+1]* ymat[a,n]
                for m in M_indices[0][a]:
                    dymat[a,b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,b]
                for c_ in range(uNum):
                    for c__ in range(uNum):
                        dymat[a,b] += dMat[a,c_,q+1]* dMat[b,c__,q+1]* np.exp(1j*T*(om_eg[a,c_]-om_eg[b,c__]))* ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode0_jit(t,N,lNum,uNum,pNum,Gamma,r,R1sum,R2sum,tswitch,M):
    dNdt = np.zeros(lNum+uNum)
    if floor(t/tswitch)%2 == 1: R_sum=R1sum
    else: R_sum=R2sum
    
    for l in prange(lNum):
        for u in prange(uNum):
            dNdt[l] += Gamma* r[l,u] * N[lNum+u] + R_sum[l,u] * (N[lNum+u] - N[l])
        if not M.size == 0:
            for k in prange(lNum):
                dNdt[l] -= M[l,k] * (N[l]-N[k])
    for u in prange(uNum):
        dNdt[lNum+u]  = -Gamma*N[lNum+u]
        for l in prange(lNum):
            dNdt[lNum+u] += R_sum[l,u] * (N[l] - N[lNum+u])
                          
    return dNdt

#%%
def ode0(t,N,lNum,uNum,pNum,Gamma,r,R1sum,R2sum,tswitch,M):
    dNdt = np.zeros(lNum+uNum)
    if floor(t/tswitch)%2 == 1: R_sum=R1sum
    else: R_sum=R2sum
    
    Nlu_matr = np.subtract.outer(N[:lNum], N[lNum:lNum+uNum])  
    
    dNdt[:lNum] = - np.sum(R_sum*Nlu_matr, axis=1) + Gamma*np.dot(r,N[lNum:lNum+uNum])
    if not M.size == 0:
        dNdt[:lNum] -= np.sum(M * np.subtract.outer(N[:lNum], N[:lNum]), axis=1)
    
    dNdt[lNum:lNum+uNum] = -Gamma*N[lNum:lNum+uNum] + np.sum(R_sum*Nlu_matr, axis=0)
              
    return dNdt

#%%
def ode1(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,w,k,r_k,m,tswitch,M,pos_dep):    
    dydt = np.zeros(lNum+uNum+3+3)
    if floor(t/tswitch)%2 == 1: rx=rx1
    else: rx=rx2
    sp = sp_.copy()
    # position dependent Force on particle due to Gaussian shape of Laserbeam:
    if pos_dep:
        for p in range(pNum):
            #if abs(y[-3]) > 1e-3: sp[p] = 1e-1
            d = np.linalg.norm(np.cross( y[-3:]-r_k[p] , k[p]/np.linalg.norm(k[p]) ))
            # r2 = np.dot(y[-3:], y[-3:]) - (np.dot(k[p], y[-3:])/np.linalg.norm(k[p]))**2
            sp[p] = sp[p] * np.exp(-2 * d**2 / ((0.2*w[p])**2) )
    
    # shape of k: (pNum,3)
    # shape of rx = (lNum,uNum,pNum), sp.shape = (pNum) ==> (rx*sp).shape = (lNum,uNum,pNum)
    # R = Gamma/2 * (rx*sp) / ( 1+4*(delta)**2/Gamma**2 )
    R = Gamma/2 * (rx*sp) / ( 1+4*( delta - np.dot(k,y[lNum+uNum:lNum+uNum+3]) )**2/Gamma**2 )    
    # sum R over pNum
    R_sum = np.sum(R,axis=2)
    # shape(Nlu_matr) = (lNum,uNum)
    Nlu_matr = y[:lNum,None] - y[None,lNum:lNum+uNum]#np.subtract.outer(y[:lNum], y[lNum:lNum+uNum])

    # __________ODE:__________
    # N_l' = ...
    dydt[:lNum] = - np.sum(R_sum*Nlu_matr, axis=1) + Gamma*np.dot(r,y[lNum:lNum+uNum])
    if not M.size == 0: # magnetic remixing of the ground states
        dydt[:lNum] -= np.sum(M * np.subtract.outer(y[:lNum], y[:lNum]), axis=1)
    # N_u' = ... 
    dydt[lNum:lNum+uNum] = -Gamma*y[lNum:lNum+uNum] + np.sum(R_sum*Nlu_matr, axis=0)
    # v' = ...
    dydt[lNum+uNum:lNum+uNum+3] = hbar/m * np.sum(np.dot(R,k) * Nlu_matr[:,:,None], axis=(0,1)) #+ g 
    # r' = ...    
    dydt[lNum+uNum+3:lNum+uNum+3+3] = y[lNum+uNum:lNum+uNum+3]
              
    return dydt

#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode1_jit(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,w,w_cylind,k,r_k,m,tswitch,M,pos_dep,beta):    
    dydt = np.zeros(lNum+uNum+3+3)
    if floor(t/tswitch)%2 == 1: rx=rx1
    else: rx=rx2
    sp = sp_.copy()
    # position dependent Force on particle due to Gaussian shape of Laserbeam:
    if pos_dep:
        if w_cylind[0] != .0 :
            for p in range(pNum):
                sp[p] = sp[p] * np.exp(-2*( (y[-3]-r_k[p,0])**2/(w_cylind[p]**2)
                                           +(y[-1]-r_k[p,2])**2/(w[p]**2)    ))
        else:
            for p in range(pNum):
                d = np.linalg.norm(np.cross( y[-3:]-r_k[p] , k[p]/np.linalg.norm(k[p]) ))
                sp[p] = sp[p] * np.exp(-2 * d**2 / ((w[p])**2) )
    delta_ = delta + 2*pi*beta*t #frequency chirping
    # shape of k: (pNum,3)
    # shape of rx = (lNum,uNum,pNum), sp.shape = (pNum) ==> (rx*sp).shape = (lNum,uNum,pNum)
    # R = Gamma/2 * (rx*sp) / ( 1+4*(delta)**2/Gamma**2 )
    R = Gamma/2 * (rx*sp) / ( 1+4*( delta_ - np.dot(k,y[lNum+uNum:lNum+uNum+3]) )**2/Gamma**2 )    
    # sum R over pNum
    R_sum = np.sum(R,axis=2)
    
    # __________ODE:__________
    # N_l' = ...
    for l in range(lNum):
        for u in range(uNum):
            dydt[l] += Gamma* r[l,u] * y[lNum+u] + R_sum[l,u] * (y[lNum+u] - y[l])
    if not M.size == 0:
        for l1 in range(lNum):
            for l2 in range(lNum):
                dydt[l1] -= M[l1,l2] * (y[l1]-y[l2])
    # N_u' = ... 
    for u in range(uNum):
        dydt[lNum+u]  = -Gamma*y[lNum+u]
        for l in range(lNum):
            dydt[lNum+u] += R_sum[l,u] * (y[l] - y[lNum+u])
    # v' = ...
    for i in range(3):
        for l in range(lNum):
            for u in range(uNum):
                for p in range(pNum):
                    dydt[lNum+uNum+i] +=  hbar/m * k[p,i] * R[l,u,p] * ( y[l] - y[lNum+u] )
    # r' = ...    
    dydt[lNum+uNum+3:lNum+uNum+3+3] = y[lNum+uNum:lNum+uNum+3]
              
    return dydt

#%%
def save_object(obj,filename=None,maxsize=20e3):
    """Save an entire class with all its attributes instead of saving a pyplot figure.
    
    Parameters
    ----------
    obj : object
        The object you want to save.
    filename : str, optional
        the filename to save the data. The extension '.pkl' will be added for
        saving the file. If no filename is provided, it is set to the attribute
        `description` of the object and if the object does not have this
        attribute, the filename is set to the name of the class belonging to
        the object.
    maxsize : int, optional
        With this option the attributes ``N``, ``t``, ``Nscatt`` and
        ``Nscattrate`` are shrunken to this value by averaging over the other variable
        entries. Usefull for preventing the files to get to big in disk space.
        The default is 20e3 which results approximately in a file size of 12MB.

    Returns
    -------
    None.
    """
    if filename == None:
        if hasattr(obj,'description'): filename = obj.description
        else: filename = type(obj).__name__ # instance is set to name of its class
    if type(obj).__name__ == 'System':
        maxs = maxsize
        if obj.t.size > 2*maxs:
            var_list = [obj.N,obj.t,obj.Nscatt,obj.Nscattrate,obj.v,obj.r]
            for i,var in enumerate(var_list):
                sh1 = int(var.shape[-1])
                n1 = int(sh1 // maxs)
                if var.ndim > 1:
                    var = var[:,:sh1-(sh1%n1)].reshape(var.shape[0],-1, n1).mean(axis=-1)
                else: var = var[:sh1-(sh1%n1)].reshape(-1, n1).mean(axis=-1)
                var_list[i] = var
            obj.N,obj.t,obj.Nscatt,obj.Nscattrate,obj.v,obj.r = var_list
        #self.N = np.array(self.N,dtype='float16')
    with open(filename+'.pkl','wb') as output:
        pickle.dump(obj,output,-1)
        
def open_object(filename):
    """Opens a saved object from a saved .pkl-file with all its attributes.    

    Parameters
    ----------
    filename : str
        filename without the '.pkl' extension.

    Returns
    -------
    output : Object
    """
    with open(filename+'.pkl','rb') as input:
        output = pickle.load(input)
    return output

#%%
def magn_remix(grs,magn_strength):
    """returns a matrix to remix all adjacent ground hyperfine levels
    by a magnetic field with certain field strength. The default is False.

    Parameters
    ----------
    grs : :class:`~Levelsystem.Groundstates`
        groundstates for which the matrix is to be build.
    magn_strength : float
        measure of the magnetic field strength (i.e. the magnetic remixing
        matrix is multiplied by 10^magn_strength). Reasonable values are
        between 6 and 9.
    
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
            elif grs[i].nu      == grs[j].nu \
                and grs[i].F    == grs[j].F \
                and grs[i].J    == grs[j].J \
                and abs(grs[i].mF-grs[j].mF) <= 1:
                matr[i,j] = 1
    return 10**(magn_strength)*matr

#%%
def selrule(gr,ex,pol):
    """Determines the selection rule between a ground state and an excited state
    caused by a certain polarization.
    
    Parameters
    ----------
    gr : :class:`~Levelsystem.Groundstate`
        Groundstate Object.
    ex : :class:`~Levelsystem.Excitedstate`
        Excited state object.
    pol : str, tuple(str, str)
        polarization of the light interacting with the two states.
        See :class:`~Lasersystem.Laser`

    Returns
    -------
    int
        either 0 or 1 for forbidden or allowed transition.
    """
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
#%%
if __name__ == '__main__':
    system = System()
    
    pol     = 'lin'
    # pol     = 'sigmap'
    # pol     = ('sigmap','sigmam')
    system.lasers.freq_pol_switch = 5e6
    system.v0 = np.array([190,0,0])
    system.r0 = np.array([0,0,0])
    
    system.lasers.add_sidebands(859.830e-9,20e-3,pol,AOM_shift=16.7e6,
                                EOM_freq=39.33e6,k=[0,1,0],r_k=[10e-3,0,0])
    # system.lasers.add_sidebands(895.699e-9,14e-3,pol,AOM_shift=16.7e6,
                                # EOM_freq=39.33e6,k=[0,1,0],r_k=[10e-3,0,0])
    # system.lasers.add_sidebands(897.961e-9,14e-3,pol,AOM_shift=16.7e6,
                                # EOM_freq=39.33e6)
    # system.lasers.add_sidebands(900.238e-9,14e-3,pol,AOM_shift=20.65e6, EOM_freq=39.33e6)
    # system.lasers.add(859.830e-9,20e-3,pol,freq_shift=-56.7660e6)
    # system.lasers.add(895.699e-9,14e-3,pol)
    # system.lasers.add(897.961e-9,14e-3,pol)
    # system.lasers.add(900.238e-9,14e-3,pol)
    
    system.levels.add_all_levels(nu_max=0)
    # system.levels.dMat.iloc[4:7,1:] *= -1
    # nodetuned_list contains tuples (nu_gr,nu_ex,p) determining the ground-/
    # excited state nu being in perfect resonance with the pth laser
    nodetuned_list = [(0,0,0),(1,0,1),(2,1,2),(3,2,3)]
    # system.N0 = [0, 0,0,0, 0,0,0, 1,0,0,0,1,  0, 0,0,0]
    system.add_magnfield(5e-4,direction=[0,1,1])
    system.levels.grstates.del_lossstate()
    system.calc_OBEs(t_int=5e-6,dt=0.1e-10,perfect_resonance=False, method='RK45',
                        nodetuned_list=nodetuned_list,magn_remixing=False,
                        velocity_dep=False,position_dep=False,calculated_by='YanGroupnew')


    