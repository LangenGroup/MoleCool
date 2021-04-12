# -*- coding: utf-8 -*-
"""
Created on Tue June 09 10:17:00 2020

@author: fkogel

v2.5.2

This module contains the main class :class:`System` which provides all information
about the Lasers, Levels and can carry out simulation calculations, e.g.
via the rate equations or Optical Bloch equations (OBEs).

Examples
--------
Example for setting up a level and laser system with the cooling and 3 repumping
lasers and calculating the dynamics::
    
    from System import *
    system = System(description='Test',load_constants='BaF')
    
    # set up the lasers each with four sidebands
    for lamb in np.array([859.830, 895.699, 897.961, 900.238])*1e-9:
        system.lasers.add_sidebands(lamb=lamb,P=20e-3,pols='lin',AOM_shift=20e6,EOM_freq=39.33e6)
    
    # include all vibrational levels up to the ground state vibrational level nu=3
    system.levels.add_all_levels(nu_max=3)
    system.levels.print_properties()
    
    # calculate dynamics with rate equations
    system.calc_rateeqs(t_int=20e-6,magn_remixing=False,method='LSODA')
    # plot populations and force
    system.plot_N(smallspacing=1e-5)
    system.plot_F()
    
Example for a Molecule with a initial velocity and position of :math:`v_x=200m/s`
and :math:`r_x=-2mm` which is transversely passing two cooling lasers with
a repumper each in the distance :math:`4mm`::
    
    from System import *
    system = System(description='Test2',load_constants='BaF')
    
    # specify initial velocity and position of the molecule
    system.v0 = np.array([200,0,0])
    system.r0 = np.array([-2e-3,0,0])
    
    # set up the cooling laser and first repumper with their wave vectors k and positions r_k
    FWHM,P = 1e-3,5e-3
    system.lasers.add_sidebands(lamb=859.830e-9,P=P,FWHM=FWHM,k=[0,1,0])
    system.lasers.add_sidebands(lamb=895.699e-9,P=P,FWHM=FWHM,k=[0,1,0])
    system.lasers.add_sidebands(lamb=859.830e-9,P=P,FWHM=FWHM,k=[0,1,0],r_k=[4e-3,0,0])
    system.lasers.add_sidebands(lamb=895.699e-9,P=P,FWHM=FWHM,k=[0,1,0],r_k=[4e-3,0,0])
    
    # include first two vibrational levels of electronic ground state and the
    # first vibrational level of the excited state
    system.levels.add_all_levels(nu_max=1)
    
    # calculate dynamics with velocity and position dependence of the laser beams and molecules
    system.calc_rateeqs(t_int=40e-6,magn_remixing=False,
                        velocity_dep=True,position_dep=True,method='LSODA')
    
    # plot scattering rate, scattered photons, velocity and position
    system.plot_Nscattrate()
    system.plot_Nscatt()
    system.plot_v()
    system.plot_r()
    system.plot_F()
"""
import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
from scipy.constants import c,h,hbar,pi,g,u,physical_constants
from scipy.constants import k as k_B
from sympy.physics.wigner import clebsch_gordan,wigner_3j,wigner_6j
from Lasersystem import *
from Levelsystem import *
import constants
from math import floor
import _pickle as pickle
import time
from numba import jit, prange
import sys, os
import multiprocessing
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.ticker as mtick
# plt.rcParams['errorbar.capsize']=3
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('figure', figsize=(7,4.35))
# plt.rcParams['figure.constrained_layout.use'] = True
# plt.rc('savefig', pad_inches=0.05, bbox='tight')
# plt.rc('axes', grid = True)
# plt.rc('grid', linestyle ='dashed', linewidth=0.5, alpha=0.7)
# plt.rc('lines', lw=1)
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
        load_constants : str, optional
            Name of a certain molecule, atom or more general system whose
            respective level constants to be loaded or imported by the class
            :class:`~Levelsystem.Levelsystem` via the constants defined in the
            module :py:class:`constants`. The default is 'BaF'.
        """
        self.lasers = Lasersystem()
        self.levels = Levelsystem(load_constants=load_constants)
        self.Bfield = Bfield()
        #self.particles = Particlesystem()
        if description == None:
            self.description = os.path.basename(sys.argv[0])[:-3]
        else:
            self.description = description
        self.N0 = np.array([]) #: initial population of all levels
        self.v0     = np.array([0.,0.,0.]) #: initial velocity of the particle
        self.r0     = np.array([0.,0.,0.]) #: initial position of the particle
        self.steadystate = {'t_ini'       : None,
                            'maxiters'    : 100,
                            'condition'   : [0.1,50],
                            'period'      : None}
        self.multiprocessing = {'processes' : multiprocessing.cpu_count()-1,#None
                                'maxtasksperchild' : None}
        print("System is created with description: {}".format(self.description))
        
    def calc_rateeqs(self,t_int=20e-6,t_start=0.,dt=None,t_eval = [],
                     perfect_resonance=False, nodetuned_list=[],
                     magn_remixing=False, magn_strength=8,
                     velocity_dep=False, position_dep=False,
                     calculated_by='YanGroupnew',verbose=True,
                     mp=False,return_fun=None,**kwargs):
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
        self.calcmethod = 'rateeqs'
        #___input arguments of this called function
        self.args = locals()
        
        #___parameters belonging to the levels
        self.levels.verbose = verbose
        Gamma       = self.levels.exstates.Gamma
        self.m      = self.levels.mass
        #branching ratios
        self.r_     = self.levels.calc_branratios(calculated_by)
        #electric dipole matrix
        self.dMat   = self.levels.calc_dMat()
        #number of ground, excited states and lasers
        lNum,uNum,pNum = self.levels.lNum, self.levels.uNum, self.lasers.pNum
        
        #___start multiprocessing if desired
        if mp:
            self.results = multiproc(obj=deepcopy(self),kwargs=self.args)
            return None
        mp_time0 = time.gmtime()[5]
        
        #___parameters belonging to the lasers  (and partially to the levels)
        self.sp     = np.array([ la.I / ( pi*c*h*Gamma/(3*la.lamb**3) ) 
                                for la in self.lasers ])
        self.k      = np.array([ la.k*la.kabs for la in self.lasers ])
        self.f      = np.array([la.f_q for la in self.lasers])
        self.w      = np.array([la.w for la in self.lasers ])
        self._w_cylind = np.array([la._w_cylind for la in self.lasers ])
        self.r_k    = np.array([ np.array(la.r_k,dtype=float) for la in self.lasers ])
        self.r_cyl_trunc = np.array([la._r_cylind_trunc for la in self.lasers])
        self.dir_cyl= np.array([la._dir_cylind for la in self.lasers],dtype=float) #unit vectors
        self.beta   = np.array([la.beta for la in self.lasers ])
        omega_k     = np.array([la.omega for la in self.lasers])
        #polarization switching time
        tswitch = 1/self.lasers.freq_pol_switch
        #calculate excitation rate R1, R2 (two for switching) with detunings and selection rules
        self.rx1    = np.zeros((lNum,uNum,pNum))
        self.rx2    = np.zeros((lNum,uNum,pNum))
        self.delta  = omega_k[None,None,:] - self.levels.calc_freq()[:,:,None]
        if np.any([la.pol_switching for la in self.lasers]):
            for l,gr in enumerate(self.levels.grstates):
                for u,ex in enumerate(self.levels.exstates):
                    for p,la in enumerate(self.lasers):
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
        else:
            self.rx1 = np.abs(np.dot(self.dMat,self.f.T))**2
            self.rx2 = self.rx1.copy()
        self.R1 = Gamma/2*self.rx1*self.sp[None,None,:] / (1+4*(self.delta-np.dot(self.k,self.v0)[None,None,:])**2/Gamma**2)
        self.R2 = Gamma/2*self.rx2*self.sp[None,None,:] / (1+4*(self.delta-np.dot(self.k,self.v0)[None,None,:])**2/Gamma**2)
        #sum R1 & R2 over pNum:
        self.R1sum, self.R2sum = np.sum(self.R1,axis=2), np.sum(self.R2,axis=2)
        
        #___magnetic remixing of the ground states. An empty array is left for no B-field 
        if magn_remixing: self.M = self.Bfield.get_remix_matrix(self.levels.grstates,remix_strength=magn_strength)
        else: self.M = np.array([[],[]])
        
        #___specify the initial (normalized) occupations of the levels
        self.N0 = np.array(self.N0, dtype=float)
        if len(self.N0) == 0:
            N0_indices = [i for i,st in enumerate(self.levels.grstates) if st.nu==0]    
            self.N0      = np.zeros(lNum+uNum)
            for i in N0_indices:
                self.N0[i] = 1.0
        self.N0 /= self.N0.sum()

        #___determine the time points at which the ODE solver should evaluate the equations
        if len(t_eval) != 0: self.t_eval = np.array(t_eval)
        else:   
            if dt != None and dt < t_int:
                self.t_eval = np.linspace(t_start,t_start+t_int,int(t_int/dt)+1)
            else:
                self.t_eval = None
        
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
            sol = solve_ivp(ode0_rateeqs_jit, (t_start,t_start+t_int), self.y0,
                    t_eval=self.t_eval, **kwargs,
                    args=(lNum,uNum,pNum,Gamma,self.r_,self.R1sum,self.R2sum,
                          tswitch,self.M))
        else:
            sol = solve_ivp(ode1_rateeqs_jit, (t_start,t_start+t_int), self.y0,
                    t_eval=self.t_eval, max_step = 10e-6, **kwargs,
                    args=(lNum,uNum,pNum,Gamma,self.r_,self.rx1,self.rx2,
                          self.delta,self.sp,self.w,self._w_cylind,
                          self.k,np.linalg.norm(self.k,axis=-1),self.r_k,
                          self.r_cyl_trunc,self.dir_cyl,
                          self.m,tswitch,self.M,position_dep,self.beta))
            #velocity v and position r
            self.v = sol.y[-6:-3]
            self.r = sol.y[-3:]
        #___execution time for the ODE solving
        self.exectime = time.perf_counter()-start_time
        if verbose: print(" execution time: {:2.4f} seconds, (t_start, t_end) = ({}s, {}s)".format(self.exectime,mp_time0,time.gmtime()[5]))
        
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
                print('WARNING: the sum of the occupations does not remain stable! Deviation: {:.2E}'.format(dev))
        if return_fun: return return_fun(self)
        
    #%%        
    def plot_all(self):
        self.plot_N(); self.plot_Nscatt(); self.plot_Nscattrate(); self.plot_Nsum()
    def plot_N(self,figname=None,figsize=(12,5),smallspacing=0.001):
        """plot populations of all levels over time."""
        if figname == None:
            plt.figure('N ({}): {}, {}, {}'.format(
                self.calcmethod,self.description,self.levels.description,
                self.lasers.description), figsize=figsize)
        else: plt.figure(figname,figsize=figsize)

        lNum, uNum = self.levels.lNum, self.levels.uNum
        lNum_red   = len([1 for st in self.levels.grstates if st.nu == 0])
        colors_l = pl.cm.jet(np.linspace(0.05,0.95,lNum_red))
        colors_u = pl.cm.jet(np.linspace(0,1,uNum))
        for i,grstate in enumerate(self.levels.grstates):
            if grstate.name == 'Loss state':
                label = 'Loss state'
                ls,color = '-',np.array([0,0,0,1])
            else:
                label='$g: J={:1.1f}, F={:1.1f}, mF={:+1.1f}$'.format(grstate.J,grstate.F,grstate.mF)
                if grstate.nu == 0: ls,color = '-',colors_l[i]
                else: ls,color = '--',[*colors_l,*colors_l,*colors_l,*colors_l,*colors_l][i]
            plt.plot(self.t*1e6,(self.N[i,:]+smallspacing*i)*1e2,
                     label=label,c=color,ls=ls)
        for i,exstate in enumerate(self.levels.exstates):
            label =   '$e: J={:1.1f}, F={:1.1f}, mF={:+1.1f}$'.format(exstate.J,exstate.F,exstate.mF)
            plt.plot(self.t*1e6,(self.N[lNum+i,:]+smallspacing*i)*1e2,
                     label=label,c=colors_u[i],ls='-.')
        
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Populations $N$ in %')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small',labelspacing=-0.0)
        # plt.tight_layout()
        
    def plot_Nscatt(self):
        """plot the scattered photon number over time (integral of `Nscattrate`)."""
        plt.figure('Nscatt: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Totally scattered photons')
        plt.plot(self.t*1e6, self.Nscatt, '-')
    
    def plot_Nscattrate(self):
        """plot the photon scattering rate over time (derivative of `Nscatt`)."""
        plt.figure('Nscattrate: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Photon scattering rate $\gamma\prime$ in MHz')
        plt.plot(self.t*1e6, self.Nscattrate*1e-6, '-')
        
    def plot_Nsum(self):
        """plot the population sum of all levels over time to ensure a small
        numerical deviation of the ODE solver."""
        plt.figure('Nsum: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Population sum $\sum N_i$')
        plt.plot(self.t*1e6, np.sum(self.N,axis=0), '-')
    
    def plot_dt(self):
        """plot the time steps at which the populations are calculated. If no `dt`
        argument is given for the calulations they are chosen from the ODE solver."""
        if 'method' in self.args['kwargs']: method = self.args['kwargs']['method']
        else: method = 'RK45'
        plt.figure('dt: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('timestep d$t$ in s')
        plt.plot(self.t[:-1]*1e6,np.diff(self.t),label=method)
        plt.yscale('log')
        plt.legend()
        
    def plot_v(self):
        """plot the velocity over time for all three axes 'x','y', and'z'."""
        plt.figure('v: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('velocity $v$ in m/s')
        ls_arr = ['-','--','-.']
        for i,axis in enumerate(['x','y','z']):
            plt.plot(self.t*1e6,self.v[i,:],label='$v_{}$'.format(axis),ls=ls_arr[i])
        plt.legend()
    
    def plot_r(self):
        """plot the position over time for all three axes 'x','y', and'z'."""
        plt.figure('r: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('position $r$ in m')
        ls_arr = ['-','--','-.']
        for i,axis in enumerate(['x','y','z']):
            plt.plot(self.t*1e6,self.r[i,:],label='$r_{}$'.format(axis),ls=ls_arr[i])
        plt.legend()
        
    def plot_FFT(self,only_sum=True,start_time=0.0):
        """plot the fast Fourier transform (FFT) of the time-dependent populations.
        
        Parameters
        ----------
        only_sum : bool, optional
            if True the sum of the FFTs of all populations is plottet. Otherwise
            the distinct FFTs for all levels are shown. The default is True.
        start_time : float between 0 and 1, optional
            starting time in units of the interaction time `t_int` at which the
            FFT is calculated. The default is 0.0.
        """
        FT_sum = 0
        t_int = self.args['t_int']
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
    
    def calc_Rabi_freqs(self,average_levels=True):
        """calculates detuning-weighted, averaged angular Rabi frequencies
        for every laser (with 2*pi included)."""
        Gamma = self.levels.exstates.Gamma
        Rabi_freqs = []
        for k,la in enumerate(self.lasers):
            Rabi_freqs_arr = np.abs(self.G[k]/np.sqrt(2)*np.dot(self.dMat, self.f[k,:])* Gamma)
            if average_levels:
                weights= np.exp(-4*(self.omega_eg-self.omega_k[k])**2/(Gamma**2))
                # protection against the case when all weights are zero because the
                # detuning is too large so that exp() gets 0.0
                if np.all(weights==0.0): weights = weights*0 + 1
                weights = weights*np.sign(Rabi_freqs_arr) * np.abs(np.dot(self.dMat, self.f[k,:]))**2
                # print(Rabi_freqs_arr,'\n',weights)
                Rabi_freqs.append( (Rabi_freqs_arr*weights).sum()/weights.sum() )
            else: 
                Rabi_freqs.append( Rabi_freqs_arr )
        self.Rabi_freqs = np.array(Rabi_freqs)
        if not average_levels: self.Rabi_freqs = np.transpose(self.Rabi_freqs,axes=(1,2,0))
        return self.Rabi_freqs
    
    def plot_F(self,figname=None):
        """plot the Force over time for all three axes 'x','y', and'z'."""
        if figname == None:
            plt.figure('F ({}): {}, {}, {}'.format(
                self.calcmethod,self.description,self.levels.description,
                self.lasers.description))
        else: plt.figure(figname,figsize=figsize)
        F = self.F/ (hbar*2*pi/860e-9*self.levels.exstates.Gamma/2)
        ls_arr = ['-','--','-.']
        for i,axis in enumerate(['x','y','z']):
            plt.plot(self.t*1e6,F[i,:],label='$F_{}$'.format(axis),ls=ls_arr[i])
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Force $F$ in $\hbar k \Gamma/2$')
        plt.legend()
    
    @property
    def F(self):
        """calculate the force over time.

        Returns
        -------
        F : np.ndarray, shape(3,ntimes)
            Force array for all <ntimes> time points and three axes 'x','y','z'.
        """
        if self.calcmethod == 'rateeqs':
            if not self.args['position_dep'] and not self.args['velocity_dep']:
                lNum,uNum = self.levels.lNum, self.levels.uNum
                N_lu = self.N[:lNum,:][:,None,:] - self.N[lNum:lNum+uNum,:][None,:,:]
                F = hbar * np.sum( np.dot(self.R1,self.k)[:,:,:,None] * N_lu[:,:,None,:], axis=(0,1)) #+ g 
            else:
                F = np.zeros((3,self.t.size))
                F[:,1:] = np.diff(self.v)/np.diff(self.t)*self.levels.mass
        if self.calcmethod == 'OBEs':
            Gamma   = self.levels.exstates.Gamma
            T       = Gamma*self.t
            # F       = np.zeros((3,self.t.size)) #size of t?
            # for k in range(self.lasers.pNum):
            #     for q in [-1,0,1]:
            #         for c in range(self.levels.lNum):
            #             for c_ in range(self.levels.uNum): #sign has to be + !!!
            #                 F += 2*np.real(hbar*Gamma*self.G[k]/2/(2**0.5) \
            #                     *np.exp(1j*self.phi[k]+1j*T*(self.om_eg[c,c_]-self.om_k[k])) \
            #                     *self.dMat[c,c_,q+1]*self.ymat[self.levels.lNum+c_,c][None,:] \
            #                     *1j*self.k[k,:][:,None] *self.f[k,q+1] )
            # F = 2*np.real(hbar*Gamma*self.G[:,None,None,None,None,None]/2/(2**0.5) \
                # *np.exp(1j*self.phi[:,None,None,None,None,None]+1j*T*(self.om_eg[None,None,:,:,None,None]-self.om_k[:,None,None,None,None,None])) *np.transpose(self.h_gek,axes=(2,0,1))[:,None,:,:,None,None]\
                # *np.transpose(self.dMat,axes=(2,0,1))[None,:,:,:,None,None]*np.transpose(self.ymat[self.levels.lNum:,:self.levels.lNum,:],axes=(1,0,2))[None,None,:,:,None,:] \
                # *1j*self.k[:,None,None,None,:,None] *self.f[:,:,None,None,None,None] ).sum(axis=(0,1,2,3))
            size    = T.size
            F       = np.zeros((3,size))
            for i,t1 in enumerate(range(0,size,size//50)):
                t2  = t1 + size//50
                if t2 > size: t2 = size
                F[:,t1:t2]  += 2*hbar*Gamma*np.real(np.transpose(self.ymat[self.levels.lNum:,:self.levels.lNum,t1:t2],axes=(1,0,2))[:,:,None,None,:] \
                                     *self.Gfd[:,:,:,None,None]*np.exp(1j*T[None,None,None,None,t1:t2]*self.om_gek[:,:,:,None,None]) \
                                     *self.k[None,None,:,:,None] ).sum(axis=(0,1,2))
        return F
    
    def Isat(self,kind='2-level'):
        """calculates saturation intensity in W/m^2 for each laser"""
        if kind == '2-level':
            return np.array([pi*c*h*self.levels.exstates.Gamma/(3*la.lamb**3)
                             for la in self.lasers ])
    #%%
    def calc_trajectory(self,t_int=20e-6,t_start=0.,dt=None,t_eval=[],
                        position_dep=False,verbose=True,force_axis=None,
                        interpol_kind='linear',**kwargs):
        """for the calculation of Monte Carlo simulations of classical particles
        which are propagated through a provided pre-calculated force profile
        to be used as interpolated function"""
        
        # self.F_profile = {'F': results[0]['F'],
        #                   'Ne' : results[0]['F'],'v':None,'I':None}
        #t handling?
        #__________________________________
        def ode_MC1D(t,y,force_axis,position_dep):
            dy      = np.zeros(6+1)
            v_proj = np.sum(y[:3]*force_axis)
            if position_dep:
                dy[:3] = a(v_proj,y[3:6])*force_axis
                dy[-1] = GNe(v_proj,y[3:6])
            else:
                dy[:3] = a(v_proj)*force_axis
                dy[-1] = GNe(v_proj)
            dy[3:6] = y[:3]
            return dy
        #__________________________________
        v0_arr = np.atleast_2d(self.v0)
        r0_arr = np.atleast_2d(self.r0)
        if isinstance(t_int, float): t_int = np.ones(v0_arr.shape[0])*t_int   
        self.sols = []
        
        if 'v' in self.F_profile:
            v = self.F_profile['v']
            if 'I' in self.F_profile: #or position_dep?
                I = self.F_profile['I']
                I_tot = self.lasers.get_intensity_func()
                
                from scipy.interpolate import RegularGridInterpolator
                a_intp  = RegularGridInterpolator((v,I), self.F_profile['F']/self.levels.mass,
                                                  method=interpol_kind,
                                                  bounds_error=False,fill_value=None)
                GNe_intp = RegularGridInterpolator((v,I), self.F_profile['Ne']*self.levels.exstates.Gamma,
                                                   method=interpol_kind,
                                                   bounds_error=False,fill_value=None)
                self.a_intp=a_intp
                def a(v,r): return a_intp(xi=(v,I_tot(r)))
                def GNe(v,r): return GNe_intp(xi=(v,I_tot(r)))
                if force_axis == '-v':
                    v0_arr2 = v0_arr.copy()
                    v0_arr2[:,0] *= 0
                    force_axis = -v0_arr2/np.linalg.norm(v0_arr2,axis=-1)[:,None]
                elif isinstance(force_axis,(list,np.ndarray)):
                    force_axis = np.atleast_2d(np.array(force_axis)/np.linalg.norm(force_axis)) +v0_arr*0
                else:
                    raise Exception('input argument <force axis> has to be given!')
                for i,v0 in enumerate(v0_arr):
                    y0 = np.array([*v0, *r0_arr[i],0.0])
                    self.sols.append(solve_ivp(ode_MC1D, (0.,t_int[i]), y0, t_eval=None,
                                               method='LSODA', args= (force_axis[i],True) ))
            else:
                from scipy.interpolate import interp1d
                a   = interp1d(v, self.F_profile['F']/self.levels.mass, kind=interpol_kind)
                GNe = interp1d(v, self.F_profile['Ne']*self.levels.exstates.Gamma, kind=interpol_kind)
                force_axis = np.array(force_axis)/np.linalg.norm(force_axis)
                for i,v0 in enumerate(v0_arr):
                    y0 = np.array([*v0, *r0_arr[i],0.])
                    self.sols.append(solve_ivp(ode_MC1D, (0.,t_int[i]), y0, t_eval=None,
                                               method='LSODA', args= (force_axis,False) ))
        elif position_dep: #Einius: here only position dependent magnetic force
            pass
        
    #%%
    def calc_OBEs(self,t_int=20e-6,t_start=0.,dt=None,t_eval = [],
                  perfect_resonance=False, nodetuned_list=[],
                  magn_remixing=False, freq_clip_TH=500, steadystate=False,
                  velocity_dep=False, position_dep=False, rounded=False,
                  calculated_by='Lucas',verbose=True,
                  mp=False,return_fun=None,**kwargs):
        
        self.calcmethod = 'OBEs'
        #___input arguments of this called function
        self.args = locals()
        
        #___parameters belonging to the levels
        self.levels.verbose = verbose
        Gamma       = self.levels.exstates.Gamma
        self.m      = self.levels.mass 
        #electric dipole matrix
        self.dMat   = self.levels.calc_dMat()
        #frequency differences between the ground and excited states
        self.omega_eg = self.levels.calc_freq()
        self.om_eg       = self.omega_eg/Gamma
        if rounded:
            self.om_eg   = np.around(self.om_eg/rounded)*rounded
        #magnetic dipole matrix of the magnetic dipole operator mu
        self.muMat = self.levels.calc_muMat()
        #indices for the magnetic zeeman sublevels. Needed for magnetic remixing
        self.M_indices = self.levels.calc_M_indices()
        #number of ground, excited states and lasers
        lNum,uNum,pNum = self.levels.lNum, self.levels.uNum, self.lasers.pNum
        
        #___start multiprocessing if desired
        if mp:
            self.results = multiproc(obj=deepcopy(self),kwargs=self.args)
            if multiprocessing.cpu_count() > 16:
                save_object(self,maxsize=1e15)
                try:
                    sys.path.append('../')
                    import subprocess, sending_email
                    hostname = subprocess.check_output('hostname').decode("utf-8")
                    sending_email.send_message('Calculation complete!','File {} at Server {}'.format(self.description,hostname))
                except:
                    pass
            return None
        mp_time0 = time.gmtime()[5]
        
        #___parameters belonging to the lasers (and partially to the levels)
        self.G      = np.array([ la.I / ( pi*c*h*Gamma/(3*la.lamb**3) ) 
                                for la in self.lasers ])**0.5
        self.f      = np.array([la.f_q for la in self.lasers])
        self.k      = np.array([ la.k*la.kabs for la in self.lasers ])
        self.phi    = np.array([la.phi for la in self.lasers])
        self.omega_k= np.array([la.omega for la in self.lasers])
        if rounded: #maybe save these variables directly as self.omega_k???
            self.om_k    = np.around(self.omega_k/Gamma/rounded)*rounded - np.around(np.dot(self.k,self.v0)/Gamma/rounded)*rounded
        else: self.om_k  = (self.omega_k-np.dot(self.k,self.v0))/Gamma
        #polarization switching time
        tswitch = 1/self.lasers.freq_pol_switch        
        #change the intensity factor G_k when Rabi frequency is provided 
        for k,la in enumerate(self.lasers):
            Rabi = la.freq_Rabi
            if Rabi != None:
                ratio = Rabi/self.calc_Rabi_freqs(average_levels=True)[k]
                self.G[k]           *= ratio
                self.lasers[k].I    *= ratio**2    
        #coefficients h to neglect highly-oscillating terms of the OBEs (with frequency threshold freq_clip_TH)
        if freq_clip_TH == 'auto':
            FWHM = np.sqrt( 1 + 2*(self.calc_Rabi_freqs(False)/Gamma)**2 ) #in units of Gamma
            h_gek  = np.where(np.abs(self.om_eg[:,:,None]-self.om_k[None,None,:]) < 8*FWHM/2, 1.0, 0.0)
            h_gege = np.where(np.abs(self.om_eg[:,:,None,None]-self.om_eg[None,None,:,:]) < 8*np.max(FWHM)/2, 1.0, 0.0)
            self.FWHM = FWHM #remove later?
        else:
            h_gek  = np.where(np.abs(self.om_eg[:,:,None]-self.om_k[None,None,:]) < freq_clip_TH, 1.0, 0.0)
            h_gege = np.where(np.abs(self.om_eg[:,:,None,None]-self.om_eg[None,None,:,:]) < freq_clip_TH, 1.0, 0.0)
        self.h_gek,self.h_gege = h_gek,h_gege #remove later?
        #___magnetic remixing of the ground states and excited states
        if magn_remixing:
            betaB  = self.Bfield.Bvec_sphbasis/(hbar*self.levels.exstates.Gamma/self.Bfield.mu_B)
        else:
            betaB  = np.array([0.,0.,0.])
        #___coefficients for new defined differential equations
        self.Gfd = h_gek * 1j*self.G[None,None,:]/2/(2**0.5)*np.exp(1j*self.phi[None,None,:]) * np.dot(self.dMat,self.f.T)
        self.om_gek = self.om_eg[:,:,None]-self.om_k[None,None,:]
        self.betamu = tuple(1j* np.dot(self.muMat[i], np.flip(betaB*np.array([-1,1,-1]))) for i in range(2))
        self.dd = h_gege * (self.dMat[:,:,None,None,:]* self.dMat[None,None,:,:,:]).sum(axis=-1)
        self.ck_indices = (tuple(np.where(self.Gfd[i,:,:] != 0.0) for i in range(lNum)),
                           tuple(np.where(self.Gfd[:,i,:] != 0.0) for i in range(uNum)))
        self.ck_indices = (tuple( np.array([i[0],i[1]]) for i in self.ck_indices[0] ),
                       tuple( np.array([i[0],i[1]]) for i in self.ck_indices[1] ))
        
        if perfect_resonance: #does not work properly --> have to be calculated before h_gek coeffs.
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
        
        #___specify the initial (normalized) occupations of the levels
        N = lNum + uNum
        N0mat = np.zeros((N,N),dtype=np.complex64)
        self.N0 = np.array(self.N0, dtype=float)
        if len(self.N0) == 0:
            N0_indices = [i for i,st in enumerate(self.levels.grstates) if st.nu==0]    
            self.N0      = np.zeros(N)
            for i in N0_indices:
                self.N0[i] = 1.0
        self.N0 /= self.N0.sum() #initial populations are always normalized
        #transform these initial values into the density matrix elements N0mat
        N0mat[(np.arange(N),np.arange(N))] = self.N0
        
        if verbose: print('Solving ode with OBEs...', end='')
        start_time = time.perf_counter()
        
        #___if steady state is wanted, multiple calculation steps of the OBEs
        #___have to be performed while the occupations between this steps are compared
        if not steadystate:
            self._evaluate(t_start, t_int, dt, N0mat)
        else:
            #___initial propagation of the equations for reaching the equilibrium region
            if self.steadystate['t_ini']:
                self.args['t_eval'] = [t_start, self.steadystate['t_ini']] #only the start and end point are important to be calculated for initial period
                self._evaluate(t_start, self.steadystate['t_ini'], dt, N0mat)
                self.args['t_eval'] = []
                t_start = self.t[-1]
                N0mat   = self.ymat[:,:,-1]
            #___specifying interaction time for the next multiple iterations to compare
            # if callable(self.steadystate['period']):
            if isinstance(self.steadystate['period'],float):
                t_int = self.steadystate['period']
            elif self.args['rounded']:
                t_int = 2*pi/(self.levels.exstates.Gamma*self.args['rounded'])
            elif self.steadystate['period'] == 'standingwave':
                if self.v0[2] != 0: #if v0==0, then t_int is not changed and thus used for int time.
                    lambda_mean = (c/(self.om_eg*Gamma/2/pi)).mean()
                    if np.any(np.abs(c/(self.om_eg*Gamma/2/pi) /lambda_mean -1)>0.1e-2 ):#percental deviation from mean
                        print('WARNING: averaging over standing wave periods might not be accurate since the wavelengths differ.')
                    period = lambda_mean/abs(self.v0[2])#/2
                    t_int = period*(t_int//period+1) # int(t_int - t_int % period)
            self._evaluate(t_start, t_int, dt, N0mat)
            t_start = self.t[-1]
            N0mat   = self.ymat[:,:,-1]
            m1      = self.N.mean(axis=1)
            # if self.steadystate['period'] == None: t_int *= 0.1
            con1, con2 = self.steadystate['condition']
            step    = 0
            for step in range(1,self.steadystate['maxiters']):
                self._evaluate(t_start, t_int, dt, N0mat)
                m2 = self.N.mean(axis=1)
                # print('diff & prop',np.all(np.abs(m1-m2)*1e2<con1),np.all(np.abs(1-m1/m2)*1e2 <con2))
                #___check if conditions for steady state are fulfilled
                if np.all(np.abs(m1-m2)*1e2 < con1) and np.all(np.nan_to_num(np.abs(1-m1/m2)*1e2,posinf=0,neginf=0) < con2):
                    break
                else:
                    m1      = m2
                    N0mat   = self.ymat[:,:,-1]
                    t_start = self.t[-1]       
            if verbose: print(' calculation steps: ',step+1)
            
        #___execution time for the ODE solving
        self.exectime = time.perf_counter()-start_time
        if verbose: print(" execution time: {:2.4f} seconds, (t_start, t_end) = ({}s, {}s)".format(self.exectime,mp_time0,time.gmtime()[5]))
        
        #___compute several physical variables using the solution of the ODE #as properties?
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
            if np.any(self.N < -1e-3): print('WARNING: population oscillations got negative')
        if return_fun: return return_fun(self)#{'N':self.N[-1,-1]}#[self.__dict__[key] for key in return_val]
        # if return_fun == True: use default function which returns force and pops?
    def _evaluate(self,t_start,t_int,dt,N0mat):
        Gamma = self.levels.exstates.Gamma
        #___determine the time points at which the ODE solver should evaluate the equations    
        if len(self.args['t_eval']) != 0:
            self.t_eval = np.array(self.args['t_eval'])
        else:
            if dt == 'auto': dt = 1/np.max(self.calc_Rabi_freqs(average_levels=True)/2/pi)/9 #1/9 of one Rabi-oscillation
            if dt != None and dt < t_int:
                self.t_eval = np.linspace(t_start,t_start+t_int,int(t_int/dt)+1)
            else:
                self.t_eval, T_eval = None, None
        if np.all(self.t_eval) != None:
            T_eval = self.t_eval * Gamma
        
        #___transform the initial density matrix N0mat in a vector
        N = self.levels.N
        N0_vec = np.zeros( N*(N+1) )
        count = 0
        for i in range(N):
            for j in range(i,N):
                N0_vec[count]   = N0mat[i,j].real
                N0_vec[count+1] = N0mat[i,j].imag
                count += 2
                
        #___depenending on the position dependence two different ODE evaluation functions are called
        velocity_dep, position_dep = self.args['velocity_dep'], self.args['position_dep']
        if velocity_dep or position_dep:
            self.y0      = N0_vec#np.array([*self.N0, *self.v0, *self.r0])
        else:
            self.y0      = N0_vec
        
        # ---------------Ordinary Differential Equation solver----------------
        # solve initial value problem of the ordinary first order differential equation with scipy
        lNum,uNum,pNum = self.levels.lNum,self.levels.uNum,self.lasers.pNum
        kwargs = self.args['kwargs']
        if not velocity_dep and not position_dep:
            # sol = solve_ivp(ode0_OBEs, (t_start*Gamma,(t_start+t_int)*Gamma),
            #                 self.y0, t_eval=T_eval, **kwargs,
            #                 args=(lNum,uNum,pNum,self.G,self.f,self.om_eg,self.om_k,
            #                       betaB,self.dMat,self.muMat,
            #                       self.M_indices,h_gek,h_gege,self.phi)) # delete?
            sol = solve_ivp(ode1_OBEs, (t_start*Gamma,(t_start+t_int)*Gamma),
                            self.y0, t_eval=T_eval, **kwargs,
                            args=(lNum,uNum,pNum, self.M_indices,
                                  self.Gfd,self.om_gek,self.betamu,self.dd))
        else:
            sol = solve_ivp(ode1_OBEs_opt2, (t_start*Gamma,(t_start+t_int)*Gamma),
                            self.y0, t_eval=T_eval, **kwargs,
                            args=(lNum,uNum,pNum, self.M_indices,
                                  self.Gfd,self.om_gek,self.betamu,self.dd,self.ck_indices))
        
        #___transform the solution vectors back to the density matrix ymat
        y_vec = sol.y
        #: solution of the time dependent density matrix elements
        self.ymat    = np.zeros((N,N,y_vec.shape[-1]),dtype=np.complex64)
        count   = 0
        for i in range(N):
            for j in range(i,N):
                self.ymat[i,j,:] = y_vec[count] + 1j* y_vec[count+1]
                count += 2     
        self.ymat    += np.conj(np.transpose(self.ymat,axes=(1,0,2))) #is diagonal remaining purely real or complex?
        self.ymat[(np.arange(N),np.arange(N))] *=0.5
        #: solution of the time dependent populations N
        self.N = np.real(self.ymat[(np.arange(N),np.arange(N))])
        #: array of the times at which the solutions are calculated
        self.t = sol.t/Gamma
            
    def add_magnfield(self,strength,direction=[0,0,1]): #old function. could be deleted?
        """old function. Was replaced by the methods in the class Bfield"""
        self.Bfield.turnon(strength=strength,direction=direction)

#%%
class Bfield:
    def __init__(self,**kwargs):
        """Class defines a magnetic field configuration and methods to turn on
        a certain field strength and direction conveniently. When initializing
        a new system via `system=System()` a magnetic field instance with zero
        field strength is directly included in this System via `system.Bfield`.
        
        Example
        -------
        >>> B1 = Bfield()   # initialize Bfield instance
        >>> B1.turnon(strength=5e-4,direction=[0,0,1],angle=60)
        >>> print(B1)       # print properties
        >>> B1.reset()      # reset magnetic field to zero.

        Parameters
        ----------
        **kwargs
            Optional keyword arguments for directly turn on a certain magnetic
            field by using these keyword arguments within the the method
            :func:`turnon` (further information).
        """
        if kwargs:  self.turnon(**kwargs)
        else:       self.reset()
        self.mu_B = physical_constants['Bohr magneton'][0]
    def turnon(self,strength=5e-4,direction=[0,0,1],angle=None,remix_strength=None):
        """Turn on a magnetic field with a certain strength and direction.

        Parameters
        ----------
        strength : float or np.ndarray, optional
            Strength in Tesla. The default is 5e-4.
        direction : list or np.ndarray with shape (3,), optional
            Direction of the magnetic field vector. Doesn't have to be given
            as normalized array. The default is [0,0,1].
        angle : float or np.ndarray, optional
            Angle in degrees at which the magnetic field vector is pointing with
            respect to the `direction` argument. The default is None.
        remix_strength : float, optional
            measure of the magnetic field strength (i.e. the magnetic remixing
            matrix is multiplied by 10^remix_strength). Reasonable values are
            between 6 and 9. The default is None.
        """
        if np.any(strength >= 10e-4):
            print('WARNING: linear Zeeman shifts are only a good approx for B<10G.')
        self.strength = strength
        self.direction = np.array(direction)
        if np.all(angle != None):
            self.angle = angle
            self.axisforangle = self.direction
            angle = angle/360*2*pi
            if not np.all(np.sin(angle) == 0.):
                v1      = self.direction
                v_perp  = np.cross(v1,np.array([0,1,0]))
                if np.all(v_perp) == 0.0: v_perp = np.cross(v1,np.array([1,0,0]))
                v1n, v_perpn = (v1*v1).sum(), (v_perp*v_perp).sum()
                alpha = np.sqrt( (v1n/(np.sin(angle)**2) - v1n)/(v_perpn**2) )
                self.direction  = v_perp + np.tensordot(alpha,v1,axes=0)
        # self.remix_strength = remix_strength
    def turnon_earth(self,vertical='z',towardsNorthPole='x'):
        """Turn on the magnetic field of the earth at Germany with a strength
        of approximately 48 uT. The vertical component is 44 uT and the horizontal
        component directing towards the North Pole is 20 uT.

        Parameters
        ----------
        vertical : str, optional
            vertical axis. Supported values are 'x', 'y' or 'z'.
            The default is 'z'.
        towardsNorthPole : str, optional
            horizontal axis directing towards the North Pole. Supported values
            are 'x', 'y' or 'z'.The default is 'x'.
        """
        axes = {'x' : 0, 'y' : 1, 'z' : 2}
        vec = np.zeros(3)
        vec[axes[vertical]]         = 44e-6
        vec[axes[towardsNorthPole]] = 20e-6
        self.turnon(strength=np.linalg.norm(vec),direction=vec)
    def reset(self):
        """Reset the magnetic field to default which is a magnetic field
        strength 0.0 and the direction [0.,0.,1.]"""
        self.strength, self.direction = 0.0, np.array([0.,0.,1.])
        if 'angle' in self.__dict__: del self.angle, self.axisforangle
        self._remix_matrix = np.array([[],[]])
    def get_remix_matrix(self,grs,remix_strength=None):
        """return a matrix to remix all adjacent ground hyperfine levels
        by a magnetic field with certain field strength. The default is False.
    
        Parameters
        ----------
        grs : :class:`~Levelsystem.Groundstates`
            groundstates for which the matrix is to be build.
        remix_strength : float
            measure of the magnetic field strength (i.e. the magnetic remixing
            matrix is multiplied by 10^remix_strength). Reasonable values are
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
        self._remix_matrix = 10**(remix_strength)*matr
        return self._remix_matrix #if remix_strength ==None: estimate it with strength & if strength=0 return empty matrix?
    
    def __str__(self):
        return str(self.__dict__)
    
    @property
    def Bvec_sphbasis(self):
        """returns the magnetic field vector in the spherical basis."""
        strength, direction = self.strength, np.array(self.direction)
        ex,ey,ez = direction.T / np.linalg.norm(direction,axis=-1)
        eps = np.array([+(ex - 1j*ey)/np.sqrt(2), ez, -(ex + 1j*ey)/np.sqrt(2)])
        eps = np.array([ -eps[2], +eps[1], -eps[0] ])
        if type(strength)   == np.ndarray: strength = strength[:,None,None]
        if type(ex)         == np.ndarray: eps = (eps.T)[None,:]
        self._Bvec_sphbasis = eps*strength#/ (hbar*self.levels.exstates.Gamma/self.mu_B)
        return self._Bvec_sphbasis
        
#%%
def multiproc(obj,kwargs):
    #___problem solving with keyword arguments
    kwargs['mp'] = False
    for kwargs2 in kwargs['kwargs']:
        kwargs[kwargs2] = kwargs['kwargs'][kwargs2]
    del kwargs['self']
    del kwargs['kwargs']
    
    #no looping through magnetic field strength or direction for rateeqs so far
    if obj.calcmethod == 'rateeqs': obj.Bfield.reset()
    
    #___expand dimensions of strength, direction, v0, r0 in order to be able to loop through them    
    if np.array(obj.Bfield.strength).ndim == 0:   strengths = [obj.Bfield.strength]
    else:                               strengths = obj.Bfield.strength
    if np.array(obj.Bfield.direction).ndim == 1:  directions = [obj.Bfield.direction]
    else:                               directions = obj.Bfield.direction    
    if obj.r0.ndim == 1:    r0_arr = obj.r0[None,:]
    else:                   r0_arr = obj.r0
    if obj.v0.ndim == 1:    v0_arr = obj.v0[None,:]
    else:                   v0_arr = obj.v0
    
    #___loop through laser objects to get to know which variables have to get
    #___iterated and how many iterations
    #--> for the dictionaries used here it'S important that the order is ensured
    #    (this is the case since python 3.6 - now (3.8))
    laser_list = []
    laser_iters_N = {}
    for l1,la in enumerate(obj.lasers):
        laser_dict = {}
        for key in ['omega','freq_Rabi','I','phi','beta','k','r_k','f_q']:
            value = la.__dict__[key]
            if (np.array(value).ndim == 1 and key not in ['k','r_k','f_q']) \
                or (np.array(value).ndim == 2 and key in ['k','r_k','f_q']): #or also with dict comprehension
                laser_dict[key] = value
                laser_iters_N[key] = len(value)
        laser_list.append(laser_dict)
    laser_iters = list(laser_iters_N.keys())
    # if kwargs['verbose']: print(laser_list,laser_iters,laser_iters_N)
    
    #___recursive function to loop through all iterable laser variables
    def recursive(_laser_iters,index):
        if not _laser_iters:
            for i,dic in enumerate(laser_list):
                for key,value in dic.items():
                    # if kwargs['verbose']: print('Laser {}: key {} is set to {}'.format(i,key,value[index[key]]))
                    obj.lasers[i].__dict__[key] = value[index[key]]
                    #or more general here: __setattr__(self, attr_name, value)
            # if kwargs['verbose']: print('b1={},b2={},b3={},b4={}'.format(b1,b2,b3,b4))
            # result_objects.append(pool.apply_async(np.sum,args=(np.arange(3),)))
            if obj.calcmethod == 'OBEs':
                result_objects.append(pool.apply_async(deepcopy(obj).calc_OBEs,kwds=(kwargs)))
            elif obj.calcmethod == 'rateeqs':
                result_objects.append(pool.apply_async(deepcopy(obj).calc_rateeqs,kwds=(kwargs)))
            # print('next evaluation..')
        else:
            for l1 in range(laser_iters_N[ _laser_iters[0] ]):
                index[_laser_iters[0]] = l1
                recursive(_laser_iters[1:],index)
    
    #___Parallelizing using Pool.apply()
    pool = multiprocessing.Pool(obj.multiprocessing['processes'],
                                maxtasksperchild=obj.multiprocessing['maxtasksperchild']) #Init multiprocessing.Pool()
    result_objects = []
    iters_dict = {'strength': len(strengths),
                  'direction': len(directions),
                  'r0':len(r0_arr),
                  'v0':len(v0_arr),
                  **laser_iters_N}
    #if v0_arr and r0_arr have the same length they should be varied at the same time and not all combinations should be calculated.
    if len(r0_arr) == len(v0_arr) and len(r0_arr) > 1: del iters_dict['v0']
    #___looping through all iterable parameters of system and laser
    for b1,strength in enumerate(strengths):
        for b2,direction in enumerate(directions):
            obj.Bfield.turnon(strength,direction)
            for b3,r0 in enumerate(r0_arr):
                obj.r0 = r0
                for b4,v0 in enumerate(v0_arr):
                    if (len(r0_arr) == len(v0_arr)) and (b3 != b4): continue
                    obj.v0 = v0
                    recursive(laser_iters,{})
                    
    if kwargs['verbose']: print('starting calculations for iterations: {}'.format(iters_dict))
    time.sleep(.5)
    # print( [r.get() for r in result_objects])
    # results = [list(r.get().values()) for r in result_objects]
    # keys = result_objects[0].get().keys() #switch this task with the one above?
    results, keys = [], []
    for r in tqdm(result_objects,smoothing=0.0):
        results.append(list(r.get().values()))
    keys = result_objects[0].get().keys() #switch this task with the one above?
    pool.close()    # Prevents any more tasks from being submitted to the pool.
    pool.join()     # Wait for the worker processes to exit.
    
    out = {}
    iters_dict = {key:value for key,value in list(iters_dict.items()) if value != 1}
    for i,key in enumerate(keys):
        first_el = np.array(results[0][i])
        if first_el.size == 1:
            out[key] = np.squeeze(np.reshape(np.concatenate(
                np.array(results,dtype=object)[:,i], axis=None), tuple(iters_dict.values())))
        else:
            out[key] = np.squeeze(np.reshape(np.concatenate(
                np.array(results,dtype=object)[:,i], axis=None), tuple([*iters_dict.values(),*(first_el.shape)])))
    
    return out, iters_dict# also here iters_dict with actual values???

                    # index = {}
                    # for l1 in range(laser_iters_N['omega']):
                    #     index['omega'] = l1  
                    #     for l2 in range(laser_iters_N['k']):
                    #         index['k'] = l2
                    #         for i,dic in enumerate(laser_list):
                    #             for key,value in dic.items():
                    #                 print('Laser {}: key {} is set to {}'.format(i,key,value[index[key]]))
                    #                 # obj.lasers[i].__dict__[key] = value[index[key]]
                    # if calling_fun == 'calc_OBEs':
                        # result_objects.append(pool.apply_async(np.sum,args=(np.arange(3),)))
                        # result_objects.append(mp_calc(obj,betaB[c1,c2,:],**kwargs)) # --> without Pool parallelization
                    # result_objects.append(pool.apply_async(deepcopy(obj).calc_OBEs,kwds=(kwargs)))
#%%
def vtoT(v,mass=157):
    """function to convert a velocity v in m/s to a temperatur in K."""
    return v**2 * 0.5*(mass*u)/k_B
def Ttov(T,mass=157):
    """function to convert a temperatur in K to a velocity v in m/s."""
    return np.sqrt(k_B*T*2/(mass*u))

#%%
@jit(nopython=True,parallel=False,fastmath=True) #original (slow) ODE form from the Fokker-Planck paper
def ode0_OBEs(T,y_vec,lNum,uNum,pNum,G,f,om_eg,om_k,betaB,dMat,muMat,M_indices,h_gek,h_gege,phi):
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
                        dymat[a,lNum+b] += 1j*G[k]*f[k,q+1]/2/(2**0.5) *h_gek[c,b,k]*np.exp(1j*phi[k]+1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[a,c]
                    for c_ in range(uNum):
                        dymat[a,lNum+b] -= 1j*G[k]*f[k,q+1]/2/(2**0.5) *h_gek[a,c_,k]*np.exp(1j*phi[k]+1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,lNum+b]
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
                            f[k,q+1] *h_gek[c,b,k]*np.exp(1j*phi[k]+1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[lNum+a,c]
                            - np.conj(f[k,q+1]) *h_gek[c,a,k]*np.exp(-1j*phi[k]-1j*T*(om_eg[c,a]-om_k[k]))* dMat[c,a,q+1]* ymat[c,lNum+b])
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
                            f[k,q+1] *h_gek[a,c_,k]*np.exp(1j*phi[k]+1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,b]
                            - np.conj(f[k,q+1]) *h_gek[b,c_,k]*np.exp(-1j*phi[k]-1j*T*(om_eg[b,c_]-om_k[k]))* dMat[b,c_,q+1]* ymat[a,lNum+c_])
                for n in M_indices[0][b]:
                    dymat[a,b] += 1j*(-1.)**q* betaB[q+1]* muMat[0][b,n,-q+1]* ymat[a,n]
                for m in M_indices[0][a]:
                    dymat[a,b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,b]
                for c_ in range(uNum):
                    for c__ in range(uNum):
                        dymat[a,b] += dMat[a,c_,q+1]* dMat[b,c__,q+1] *h_gege[a,c_,b,c__]*np.exp(1j*T*(om_eg[a,c_]-om_eg[b,c__]))* ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode1_OBEs_opt1 but further optimized by rearranging the loops
def ode1_OBEs_opt2(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd,ck_indices):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(uNum):
        for c,k in zip(ck_indices[1][a][0],ck_indices[1][a][1]):
            for b in range(lNum):
                dymat[b,lNum+a] += Gfd[c,a,k]* np.exp(1j*om_gek[c,a,k]*T)* ymat[b,c]
            for b in range(a,uNum):            
                dymat[lNum+a,lNum+b] += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)* ymat[c,lNum+b]
        for n in M_indices[1][a]:
            for b in range(lNum):
                dymat[b,lNum+a] += betamu[1][a,n] * ymat[b,lNum+n]
        for m in M_indices[1][a]:
            for b in range(a,uNum):
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
        for b in range(a,uNum):
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for b in range(uNum-1,-1,-1):       
        for c,k in zip(ck_indices[1][b][0],ck_indices[1][b][1]):
            for a in range(0,b+1):
                dymat[lNum+a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[lNum+a,c]

    for a in range(lNum):
        for c_,k in zip(ck_indices[0][a][0],ck_indices[0][a][1]):
            for b in range(uNum):
                dymat[a,lNum+b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,lNum+b]
            for b in range(a,lNum):    
                dymat[a,b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,b]
        for m in M_indices[0][a]:
            for b in range(uNum):
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            for b in range(a,lNum):    
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
        for b in range(uNum):
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]            
        for b in range(a,lNum):
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__]
    for b in range(lNum-1,-1,-1):
        for c_,k in zip(ck_indices[0][b][0],ck_indices[0][b][1]):
            for a in range(0,b+1):
                dymat[a,b] -= np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)* ymat[a,lNum+c_]
                
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode1_OBEs but in optimized form with ck_indices variable
def ode1_OBEs_opt1(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd,ck_indices):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
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
            for c,k in zip(*ck_indices[1][b]):
                dymat[a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[a,c]
            for c_,k in zip(*ck_indices[0][a]):
                dymat[a,lNum+b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,lNum+b]
            for n in M_indices[1][b]:
                dymat[a,lNum+b] += betamu[1][b,n] * ymat[a,lNum+n]
            for m in M_indices[0][a]:
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for c,k in zip(*ck_indices[1][b]):
                dymat[lNum+a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[lNum+a,c]
            for c,k in zip(*ck_indices[1][a]):
                dymat[lNum+a,lNum+b] += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)* ymat[c,lNum+b]
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            for m in M_indices[1][a]:
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for c_,k in zip(*ck_indices[0][a]):
                dymat[a,b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,b]
            for c_,k in zip(*ck_indices[0][b]):
                dymat[a,b] -= np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)* ymat[a,lNum+c_]
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for m in M_indices[0][a]:
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode0_OBEs but in optimized form with less input variables
def ode1_OBEs(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
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
            for c in range(lNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)
                dymat[a,lNum+b] += tmp* ymat[a,c]
            for c_ in range(uNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)
                dymat[a,lNum+b] -= tmp* ymat[lNum+c_,lNum+b]
            for n in M_indices[1][b]:
                dymat[a,lNum+b] += betamu[1][b,n] * ymat[a,lNum+n]
            for m in M_indices[0][a]:
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for c in range(lNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)
                dymat[lNum+a,lNum+b] += tmp* ymat[lNum+a,c]
            for c in range(lNum):
                tmp = 0
                for k in range(pNum):
                    tmp += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)
                dymat[lNum+a,lNum+b] += tmp* ymat[c,lNum+b]
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            for m in M_indices[1][a]:
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for c_ in range(uNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)
                dymat[a,b] -= tmp* ymat[lNum+c_,b]
            for c_ in range(uNum):
                tmp = 0
                for k in range(pNum):
                    tmp += np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)
                dymat[a,b] -= tmp* ymat[a,lNum+c_]
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for m in M_indices[0][a]:
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__]
    
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
def ode0_rateeqs_jit(t,N,lNum,uNum,pNum,Gamma,r,R1sum,R2sum,tswitch,M):
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
def ode0_rateeqs(t,N,lNum,uNum,pNum,Gamma,r,R1sum,R2sum,tswitch,M):
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
def ode1_rateeqs(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,w,k,r_k,m,tswitch,M,pos_dep):    
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
def ode1_rateeqs_jit(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,w,w_cyl,k,kabs,r_k,r_cyl_trunc,dir_cyl,m,tswitch,M,pos_dep,beta):    
    dydt = np.zeros(lNum+uNum+3+3)
    if floor(t/tswitch)%2 == 1: rx=rx1
    else: rx=rx2
    sp = sp_.copy()
    # position dependent Force on particle due to Gaussian shape of Laserbeam:
    if pos_dep:
        for p in range(pNum):
            r_ = y[-3:] - r_k[p]
            if w_cyl[p] != 0.0: # calculation for a beam which is widened by a cylindrical lens
                d2_w = np.dot(dir_cyl[p],r_)**2
                if d2_w > r_cyl_trunc[p]**2: #test if position is larger than the truncation radius along the dir_cyl direction
                    sp[p] = 0.0  
                else:
                    d2 = np.dot(np.cross(dir_cyl[p],k[p]/kabs[p]),r_)**2
                    sp[p] = sp[p] *np.exp(-2*(d2_w/w_cyl[p]**2 + d2/w[p]**2))
            else: 
                r_perp = np.cross( r_ , k[p]/kabs[p] )
                sp[p] = sp[p] *np.exp(-2 * np.dot(r_perp,r_perp) / w[p]**2 )  

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
    """
    if filename == None:
        if hasattr(obj,'description'): filename = obj.description
        else: filename = type(obj).__name__ # instance is set to name of its class
    if type(obj).__name__ == 'System':
        if 'return_fun' in obj.args:
            del obj.args['return_fun'] #problem when an external function is tried to be saved
        if 't' in obj.__dict__:
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
def selrule(gr,ex,pol):
    """Determines the selection rule between a ground state and an excited state
    caused by a certain polarization.
    
    Parameters
    ----------
    gr : :class:`~Levelsystem.Groundstate`
        Groundstate Object.
    ex : :class:`~Levelsystem.Excitedstate`
        Excitedstate object.
    pol : str
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
    
    # nodetuned_list contains tuples (nu_gr,nu_ex,p) determining the ground-/
    # excited state nu being in perfect resonance with the pth laser
    nodetuned_list = [(0,0,0),(1,0,1),(2,1,2),(3,2,3)]
    # system.N0 = [0, 0,0,0, 0,0,0, 1,0,0,0,1,  0, 0,0,0]
    
    system.add_magnfield(5e-4,direction=[0,1,1])
    system.levels.grstates.del_lossstate()
    system.calc_OBEs(t_int=5e-6,dt=1e-9,perfect_resonance=False, method='RK45',
                        nodetuned_list=nodetuned_list,magn_remixing=False,
                        velocity_dep=False,position_dep=False,calculated_by='YanGroupnew')

    