# -*- coding: utf-8 -*-
"""
This module contains the main class :class:`~MoleCool.System.System` which provides all
information about the lasers light fields, the atomic or molecular level structure,
and the magnetic field to carry out simulation calculations, e.g.
via the rate equations or Optical Bloch equations (OBEs).
"""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.constants import c,h,hbar,pi,g,physical_constants
from scipy.constants import k as k_B
from scipy.constants import u as u_mass
from sympy.physics.wigner import clebsch_gordan,wigner_3j,wigner_6j
from MoleCool.Lasersystem import *
from MoleCool.Levelsystem import *
from MoleCool import tools
from MoleCool.tools import save_object, open_object, ODEs, return_fun_default
from MoleCool.Bfield import Bfield
import time
import sys, os
from copy import deepcopy
import multiprocessing
from tqdm import tqdm
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.ticker as mtick

np.set_printoptions(precision=4,suppress=True)
#%%
class System:
    def __init__(self, description=None,load_constants='',verbose=True):
        """An instance of this class is the starting point for simulating any
        atomic or molecular dynamics simulations. Specficially, it defines an
        object to not only store all important information about the system
        but also to calculate any time evolution.
    
        Parameters
        ----------
        description : str, optional
            A short description of this System can be added. If not provided,
            the attribute is set to the name of the respective executed main
            python file.
        load_constants : str, optional
            File name of a certain molecule, atom or more general system whose
            respective level constants to be loaded or imported by the class
            :class:`~MoleCool.Levelsystem.Levelsystem` via the constants
            defined in the .json file. The default is ''.
    
        Example
        -------
        After initiating a :class:`~MoleCool.System.System` object, the
        instances of :class:`~MoleCool.Lasersystem.Lasersystem`,
        :class:`~MoleCool.Levelsystem.Levelsystem`,
        and :class:`~MoleCool.Bfield.Bfield` can be accessed via::
            
            system = System()
            print(system.lasers)
            print(system.levels)
            print(system.Bfield)
        """
        self.lasers = Lasersystem()
        self.levels = Levelsystem(load_constants=load_constants,verbose=verbose)
        self.Bfield = Bfield()
        #self.particles = Particlesystem()
        if description == None:
            self.description = os.path.basename(sys.argv[0])[:-3]
        else:
            self.description = description
        self.reset_N0()                    #  initial population of all levels
        self.v0     = np.array([0.,0.,0.]) #: initial velocity of the particle
        self.r0     = np.array([0.,0.,0.]) #: initial position of the particle
        """dictionary for parameters specifying the steady state conditions."""
        self.steadystate = {'t_ini'       : None,
                            'maxiters'    : 100,
                            'condition'   : [0.1,50],
                            'period'      : None}
        """dictionary for parameters specifying the multiprocessing of calculations."""
        self.multiprocessing = {'processes' : int(multiprocessing.cpu_count()*0.95),#None
                                'maxtasksperchild' : None,
                                'show_progressbar' : True,
                                'savetofile'       : True}
        if verbose:
            print("System is created with description: {}".format(self.description))
        
    def calc_rateeqs(self,t_int=20e-6,t_start=0.,dt=None,t_eval = [],
                     magn_remixing=False, magn_strength=8,
                     position_dep=False, trajectory=False,
                     verbose=True, return_fun=return_fun_default,
                     **kwargs):
        """Calculate the time evolution of the single level populations with
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
        magn_remixing : bool, optional
            if True, the adjacent ground hyperfine levels are perfectly mixed
            by a magnetic field. The default is False.
        magn_strength : float, optional
            measure of the magnetic field strength (i.e. the magnetic remixing
            matrix is multiplied by 10^magn_strength). Reasonable values are
            between 6 and 9. The default is 8.
        position_dep : bool, optional
            determines whether to take the Gaussian intensity distribution of
            the laser beams into account. The default is False.
        trajectory : bool, optional
            determines whether a trajectory of the molecule is calculated using
            simple equations of motion. This yields the additional time-dependent
            parameters ``v`` and ``r`` for the velocity and position. So, the force
            which is acting on the molecule changes the velocity which in turn can
            alter the Doppler shift. Further, the `position_dep` parameter determines
            if either a uniform unitensity or complex intensity distribution due to
            the Gaussian beam shapes is assumed through which the particle is
            propagated. The default is False.
        verbose : bool, optional
            whether to print additional information like execution time or the
            scattered photon number. The default is True.
        return_fun : function-type, optional
            if `mp` == True, the returned dictionary of this function determines
            the quantities which are save for every single parameter configuration.
            The default is None.
        **kwargs : keyword arguments, optional
            other options of the `solve_ivp` scipy function can be specified
            (see homepage of scipy for further information).
        
        Note
        ----
        function creates attributes 
        
        * ``N`` : solution of the time dependent populations N,
        * ``Nscatt`` : time dependent scattering number,
        * ``Nscattrate`` : time dependent scattering rate,
        * ``photons``: totally scattered photons
        * ``args``: input arguments of the call of this function
        * ``t`` : times at which the solution was calculated
        * ``v`` : calculated velocities of the molecule at times ``t``
          (only given if `trajectory` == True)
        * ``r`` : calculated positions of the molecule at times ``t``
          (only given if `trajectory` == True)
        """
        
        self.calcmethod = 'rateeqs'
        #___input arguments of this called function
        self.args = locals()
        self.check_config(raise_Error=True)
        
        #___parameters belonging to the levels
        self.levels.calc_all()
        
        #___start multiprocessing if desired only after calculating the levels
        #   properties so that they don't have to be re-calculated every time.
        if self._identify_iter_params() or self.lasers._identify_iter_params()[0]:
            return self._start_mp()
        
        #___parameters belonging to the lasers  (and partially to the levels)
        # wave vector k
        self.k      = self.lasers.getarr('k')*self.lasers.getarr('kabs')[:,None] #no unit vectors
        self.delta  = self.lasers.getarr('omega')[None,None,:] - self.levels.calc_freq()[:,:,None]
        # saturation parameter of intensity (lNum,uNum,pNum)
        self.sp     = self.lasers.getarr('I')[None,None,:]/(self.levels.Isat[:,:,None])
        #polarization switching time
        tswitch     = 1/self.lasers.freq_pol_switch
        self.rx1    = np.abs(np.dot(self.levels.calc_dMat(),self.lasers.getarr('f_q').T))**2
        # for polarization switching:
        if np.any([la.pol_switching for la in self.lasers]):
            self.rx2 = np.abs(np.dot(self.levels.calc_dMat(),self.lasers.getarr('f_q2').T))**2
        else:
            self.rx2 = self.rx1.copy()
        
        #___magnetic remixing of the ground states. An empty array is left for no B-field 
        if magn_remixing:
            self.M = self.Bfield.get_remix_matrix(self.levels.grstates[0],remix_strength=magn_strength)
        else:
            self.M = np.array([[],[]])
        
        #___specify the initial (normalized) occupations of the levels
        self.initialize_N0()

        #___determine the time points at which the ODE solver should evaluate the equations
        if len(t_eval) != 0: self.t_eval = np.array(t_eval)
        else:   
            if dt != None and dt < t_int:
                self.t_eval = np.linspace(t_start,t_start+t_int,int(t_int/dt)+1)
            else:
                self.t_eval = None
        
        #___depenending on the position dependence two different ODE evaluation functions are called
        if trajectory:
            self.y0      = np.array([*self.N0, *self.v0, *self.r0])
        else:
            # position dependent intensity due to Gaussian shape of Laserbeam:
            if position_dep:
                self.sp *= self.lasers.I_tot(self.r0,sum_lasers=False,use_jit=False)[None,None,:]
            self.R1 = self.levels.calc_Gamma()[None,:,None]/2*self.rx1*self.sp / (
                1+4*(self.delta-np.dot(self.k,self.v0)[None,None,:])**2/self.levels.calc_Gamma()[None,:,None]**2)
            self.R2 = self.levels.calc_Gamma()[None,:,None]/2*self.rx2*self.sp / (
                1+4*(self.delta-np.dot(self.k,self.v0)[None,None,:])**2/self.levels.calc_Gamma()[None,:,None]**2)
            #sum R1 & R2 over pNum:
            self.R1sum, self.R2sum = np.sum(self.R1,axis=2), np.sum(self.R2,axis=2)
            
            self.y0      = self.N0
        
        #number of ground, excited states and lasers
        lNum,uNum,pNum = self.levels.lNum, self.levels.uNum, self.lasers.pNum
        
        # ---------------Ordinary Differential Equation solver----------------
        #solve initial value problem of the ordinary first order differential equation with scipy
        if verbose: print('Solving ode with rate equations...', end=' ')
        start_time = time.perf_counter()
        if not trajectory:
            sol = solve_ivp(ODEs.ode0_rateeqs_jit, (t_start,t_start+t_int), self.y0,
                    t_eval=self.t_eval, **self.args['kwargs'],
                    args=(lNum,uNum,pNum,self.levels.calc_Gamma(),self.levels.calc_branratios(),
                          self.R1sum,self.R2sum,tswitch,self.M))
        else:
            sol = solve_ivp(ODEs.ode1_rateeqs_jit, (t_start,t_start+t_int), self.y0,
                    t_eval=self.t_eval, **self.args['kwargs'],
                    args=(lNum,uNum,pNum,np.reshape(self.levels.calc_Gamma(),(1,-1,1)),self.levels.calc_branratios(),
                          self.rx1,self.rx2,self.delta,self.sp,
                          self.lasers.getarr('w'),self.lasers.getarr('_w_cylind'),
                          self.k,self.lasers.getarr('kabs'),self.lasers.getarr('r_k'),
                          self.lasers.getarr('_r_cylind_trunc'),self.lasers.getarr('_dir_cylind'), #unit vectors
                          self.levels.mass,tswitch,self.M,position_dep,self.lasers.getarr('beta')))
            #velocity v and position r
            self.v = sol.y[-6:-3]
            self.r = sol.y[-3:]

        #: execution time for the ODE solving
        self.exectime = time.perf_counter()-start_time
        #: array of the times at which the solutions are calculated
        self.t = sol.t
        #: solution of the time dependent populations N
        self.N = sol.y[:lNum+uNum]
        self._verify_calculation()    
        if return_fun == True: return {'system':self}
        elif return_fun: return return_fun(self)   
        
    #%%        
    def plot_all(self):
        self.plot_N(); self.plot_Nsum(); self.plot_dt()
        self.plot_Nscatt(); self.plot_Nscattrate(); self.plot_F()
        if ('trajectory' in self.args) and self.args['trajectory']:
            self.plot_r(); self.plot_v()
        
    def plot_N(self,figname=None,figsize=(12,5),smallspacing=0):
        """plot populations of all levels over time."""
        if figname == None:
            plt.figure('N ({}): {}, {}, {}'.format(
                self.calcmethod,self.description,self.levels.description,
                self.lasers.description), figsize=figsize)
        else: plt.figure(figname,figsize=figsize)
        
        N_sum = 0
        for i,ElSt in enumerate(self.levels.electronic_states):
            if 'v' in ElSt.states[0].QuNrs:
                alphas  = np.linspace(1,0,ElSt.v_max+2)[:-1] #####
                j_v0    = np.array([jj for jj,st in enumerate(ElSt) if st.v == 0])
                N_red   = len(j_v0)
            else:
                alpha   = 1.0
            N = ElSt.N
            for j, state in enumerate(ElSt.states):
                ls = ['solid','dashed','dashdot','dotted','dashdotdotted',
                      'loosely dotted','loosely dashed','loosely dashdotted'][i]
                if N > 10:
                    if 'v' in state.QuNrs:
                        if j in j_v0:
                            color   = pl.cm.jet(np.argwhere(j_v0==j)[0][0]/(N_red-1))
                        else:
                            color   = 'grey'
                    else:
                        color = pl.cm.jet(j/(N-1))
                else:
                    color = 'C' + str(j)
                    
                if 'v' in state.QuNrs:
                    if state.is_lossstate:
                        color = 'k'
                        alpha = 1.0
                    else:
                        alpha = alphas[state.v]
                
                label = str(state).split(f'{ElSt.gs_exs}=')[-1]
                    
                plt.plot(self.t*1e6,(self.N[j+N_sum,:]+smallspacing*i)*1e2,
                         label=label,ls=ls,color=color,alpha=alpha)
            N_sum += N
            
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Populations $N$ in %')
        leg = plt.legend(title='States:',loc='center left',labelspacing=-0.0,
                         bbox_to_anchor=(1, 0.5),fontsize='x-small')
        # set the linewidth of each legend object
        for legobj in leg.legend_handles:
            legobj.set_linewidth(1.4)
        
    def plot_Nscatt(self,sum_over_ElSts=False):
        """plot the scattered photon number over time (integral of `Nscattrate`)."""
        plt.figure('Nscatt: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Totally scattered photons')
        Nscatt = self.get_Nscatt(sum_over_ElSts=sum_over_ElSts)
        for i,ElSt in enumerate(self.levels.exstates):
            plt.plot(self.t*1e6, Nscatt[i,:], '-',label=ElSt.label)
        if Nscatt.shape[0]>1:
            plt.plot(self.t*1e6, np.sum(Nscatt,axis=0), '-',label='Sum')
        plt.legend()
    
    def plot_Nscattrate(self,sum_over_ElSts=False):
        """plot the photon scattering rate over time (derivative of `Nscatt`)."""
        plt.figure('Nscattrate: {}, {}, {}'.format(self.description,self.levels.description,self.lasers.description))
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Photon scattering rate $\gamma\prime$ in MHz')
        Nscattrate = self.get_Nscattrate(sum_over_ElSts=sum_over_ElSts)
        for i,ElSt in enumerate(self.levels.exstates):
            plt.plot(self.t*1e6, Nscattrate[i,:]*1e-6, '-',label=ElSt.label)
        if Nscattrate.shape[0]>1:
            plt.plot(self.t*1e6, np.sum(Nscattrate,axis=0)*1e-6, '-',label='Sum')
        plt.legend()
        
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
    
    def calc_Rabi_freqs(self,position_dep=False):
        """Calculate the (angular) pure Rabi frequencies for each transition
        and each laser component (with 2*pi included).

        Parameters
        ----------
        position_dep : bool, optional
            Whether the intensity for the calculation is evaluated at a certain
            position within the Gaussian laser beam profiles (True) or at the
            maximum (False). The default is False.

        Returns
        -------
        np.ndarray((lNum,uNum,pNum))
            Angular Rabi frequencies for each combination of laser, excited states,
            and ground state.
        """
        # saturation parameter of intensity (lNum,uNum,pNum)
        self.sp = self.lasers.getarr('I')[None,None,:]/(self.levels.Isat[:,:,None])
        self.Rabi_freqs = self.levels.calc_Gamma()[None,:,None]*np.sqrt(self.sp/2) \
                        * np.dot(self.levels.calc_dMat(), self.lasers.getarr('f_q').T)
        
        # position dependent intensity due to Gaussian shape of Laserbeam:
        if position_dep:
            self.Rabi_freqs *= np.sqrt(self.lasers.I_tot(self.r0,sum_lasers=False,use_jit=False))[None,:]
        
        # for simple levelsystems, the Rabi frequency can be set via the laser instances
        Rabi_set = self.lasers.getarr('freq_Rabi') #this feature must be tested!?
        if not np.all(np.isnan(Rabi_set)):
            Rabi_freqs_max = np.abs(self.Rabi_freqs).max(axis=(0,1)) #shape (pNum)
            for p,la in enumerate(self.lasers):
                if not np.isnan(Rabi_set[p]):
                    ratio = Rabi_set[p]/Rabi_freqs_max[p]
                    self.Rabi_freqs[:,:,p]  *= ratio
                    la.I                    *= ratio**2   
        
        return self.Rabi_freqs
    
    def plot_F(self,figname=None,axes=['x','y','z']):
        """plot the Force over time for all three axes 'x','y', and'z'."""
        if figname == None:
            plt.figure('F ({}): {}, {}, {}'.format(
                self.calcmethod,self.description,self.levels.description,
                self.lasers.description))
        else: plt.figure(figname)
        ls_arr = ['-','--','-.']
        for axis in axes:
            i = {'x':0,'y':1,'z':2}[axis]
            plt.plot(self.t * 1e6, self.F[i,:] / self.hbarkG2,
                     label='$F_{}$'.format(axis),ls=ls_arr[i])
        plt.xlabel('time $t$ in $\mu$s')
        plt.ylabel('Force $F$ in $\hbar k \Gamma_{}/2$'.format(self.levels.exstates_labels[0]))
        plt.legend()
        
    def plot_spectrum(self, wavelengths=[], lasers=True, transitions=True,
                      unit = 'MHz',
                      exs = [],
                      relative_to_wavelengths = False,
                      axs = [],
                      subplot_sep = 200, 
                      laser_spectrum_kwargs = dict(),
                      transitions_kwargs = dict(),
                      ):
        """Plot the spectrum of :class:`~.Lasersystem.Laser` objects
        and transition spectra with their respective intensities.
        This method cleverly combines :meth:`~.Lasersystem.plot_spectrum`
        and :meth:`.Levelsystem.plot_transition_spectrum` methods.

        Parameters
        ----------
        wavelengths : list, optional
            wavelengths that should be plotted within the range ``subplot_sep``.
            By default all available laser wavelengths are used.
        lasers : bool, optional
            Whether to include the laser spectrum. The default is True.
        transitions : bool, optional
            Whether to include the transition spectrum. The default is True.
        unit : str, optional
            Unit of the x-axis to be plotted.
            Can be one of ``['GHz','MHz','kHz','Hz','Gamma']``. Default is 'MHz'.
        exs : list(str), optional
            See ``exs`` in :meth:`.Levelsystem.plot_transition_spectrum`.
        relative_to_wavelengths : bool, optional
            Whether the x-axis should be plotted in absolute frequency units or
            relative to ``wavelengths``.
        axs : list of ``matplotlib.pyplot.axis`` objects, optional
            axis objects to put the plot(s) on. The default is [].
        subplot_sep : float, optional
            Defines the range of the plotted x-axis and the separation for the 
            automatic inclusion of all wavlengths (see parameter ``wavelengths``)
            in units of ``ElectronicState.Gamma``. Default is 200.
        laser_spectrum_kwargs : kwargs, optional
            Additional keyword arguments
            (see :meth:`.Lasersystem.plot_spectrum`). The default is dict().
        transitions_kwargs : kwargs, optional
            Additional keyword arguments
            (see :meth:`.Levelsystem.plot_transition_spectrum`). The default is dict().

        Returns
        -------
        axs : list of ``matplotlib.pyplot.axis``.
            Axes of the subplot(s).
        """
        if (not lasers) and (not transitions):
            raise Exception("Either one of <lasers> or <transitions> must be True!")
        
        # exctract Gamma
        if not 'exs' in transitions_kwargs:
            transitions_kwargs['exs'] = [self.levels.exstates_labels[0]]
        ExSt    = transitions_kwargs['exs'][0]
        Gamma   = self.levels[ExSt].Gamma*1e6
        
        if lasers and self.lasers.entries:
            subplot_sep_las = subplot_sep*Gamma
            if not wavelengths:
                wavelengths = self.lasers._get_wavelength_regimes(subplot_sep_las)
            std         = laser_spectrum_kwargs.pop('std', Gamma/2)
            axs_lasers  = self.lasers.plot_spectrum(
                axs = axs,
                wavelengths = wavelengths,
                unit = unit,
                relative_to_wavelengths = relative_to_wavelengths,
                subplot_sep = subplot_sep_las,
                std = std,
                **laser_spectrum_kwargs,
                )
        
            if transitions:
                axs = [ax.twinx() for ax in axs_lasers]
                for ax in axs_lasers:
                    ax.tick_params(axis='y', labelcolor='grey')
                    ax.yaxis.label.set_color('grey')
            else:
                axs = axs_lasers

        if transitions and self.levels.exstates and self.levels.grstates and self.levels.states:
            kwargs_sum = transitions_kwargs.pop(
                'kwargs_sum', dict(color='k',alpha=0.7,ls='--'))
            
            self.levels.plot_transition_spectrum(
                ax = axs,
                wavelengths = wavelengths,
                E_unit = unit,
                relative_to_wavelengths = relative_to_wavelengths,
                subplot_sep = subplot_sep,
                kwargs_sum  = kwargs_sum,
                **transitions_kwargs,
                )
        
        return axs
    
    @property
    def hbarkG2(self):
        """Calculate :math:`\hbar k \Gamma /2` using the wave vector defined
        in the laser system and natural lifetime defined in the first excited
        electronic state."""
        lambs   = self.lasers.getarr('lamb')
        dev     = 1e-3
        diff    = lambs.std()/lambs.mean()
        Gamma   = self.levels.exstates[0].Gamma*2*pi*1e6
        if len(self.levels.exstates) > 1:
            warnings.warn("For calculation of hbar * k * Gamma/2, Gamma of the first ElSt is used")
        if diff > dev:
            warnings.warn(
                ("For calculation of hbar * k * Gamma/2, the wavelengths "
                 f"of the lasers' wavelengths differ by {diff*1e2:.2f} % "
                 f"(which is more than {dev*1e2:.2f} %). The "
                 "returned unit might thus me inappropriate.")
                )
        
        return hbar*2*pi/lambs.mean()*Gamma/2
    
    @property
    def F(self):
        """calculate the force over time.

        Returns
        -------
        F : np.ndarray, shape(3,ntimes)
            Force array for all <ntimes> time points and three axes 'x','y','z'.
        """
        if self.calcmethod == 'rateeqs':
            if not self.args['trajectory']:
                lNum,uNum = self.levels.lNum, self.levels.uNum
                N_lu = self.N[:lNum,:][:,None,:] - self.N[lNum:lNum+uNum,:][None,:,:]
                F = hbar * np.sum( np.dot(self.R1,self.k)[:,:,:,None] * N_lu[:,:,None,:], axis=(0,1)) #+ g 
            else:
                F = np.zeros((3,self.t.size))
                F[:,1:] = np.diff(self.v)/np.diff(self.t)*self.levels.mass
        if self.calcmethod == 'OBEs':
            T       = self.freq_unit*self.t
            T_size  = T.size
            size    = self.levels.lNum*self.levels.uNum*self.lasers.pNum*3*T_size #total size of the force array below
            T_step  = T_size//(size//int(10e6) + 1)
            
            F       = np.zeros((3,T_size))
            for i,t1 in enumerate(range(0,T_size,T_step)): #ensure that not too much memory is needed for huge np arrays
                t2  = t1 + T_step
                if t2 > T_size: t2 = T_size
                F[:,t1:t2]  += np.real(np.transpose(self.ymat[self.levels.lNum:,:self.levels.lNum,t1:t2],axes=(1,0,2))[:,:,None,None,:] \
                                     *self.Gfd[:,:,:,None,None]*np.exp(1j*T[None,None,None,None,t1:t2]*self.om_gek[:,:,:,None,None]) \
                                     *self.k[None,None,:,:,None] ).sum(axis=(0,1,2))
            F *= 2*hbar*self.freq_unit
        
        return F
    
    #%%
    def calc_trajectory(self,F_profile,t_int=20e-6,t_start=0.,dt=None,t_eval=None,
                        verbose=True,force_axis=None,
                        interpol_kind='linear',save_scipy_sols=False,**kwargs):
        """Calculate Monte Carlo simulations of classical particles
        which are propagated through a provided pre-calculated force profile
        to be used as interpolated function"""
        
        if 'method' not in kwargs:
            kwargs['method'] = 'LSODA' # setting default solver method for ODE
        
        # Checking F_profile argument for datatype, etc.:
        if not isinstance(F_profile, dict):
            if np.all([isinstance(dic,dict) for dic in F_profile]):
                F_profile = {k:v for dic in F_profile for k,v in dic.items()}
            else:
                raise ValueError(f"F_profile must be dictionary or iterable of dictionaries!")
                
        if ('a' not in F_profile) and ('F' in F_profile):
            F_profile['a'] = F_profile['F']/self.levels.mass
            print(f'Converting force to acceleration with mass {self.levels.mass}')
        else:
            raise ValueError("Either 'F' or 'a' must be included in F_profile!")
        
        # attribute for storing the solutions of the trajectory simulations:
        self.trajectory_results = dict(kwargs=locals(), sols=[])
        
        v0_arr = np.atleast_2d(self.v0)
        r0_arr = np.atleast_2d(self.r0)
        if isinstance(t_int, float):
            t_int = np.ones(v0_arr.shape[0])*t_int   
        
        self.trajectory_results['final_values'] = dict(photons = np.zeros(len(v0_arr)),
                                                       v = np.zeros((len(v0_arr),3)),
                                                       r = np.zeros((len(v0_arr),3)))
        
        if 'v' in F_profile or 'v0' in F_profile:
            v = F_profile['v'] if 'v' in F_profile else F_profile['v0']
        else:
            raise ValueError("Either 'v' or 'v0' must be included in F_profile!")
            
        if 'I' in F_profile:
            position_dep    = True
            I               = F_profile['I']
            I_tot           = self.lasers.get_intensity_func()
            
            from scipy.interpolate import RegularGridInterpolator as interp
            a_intp  = interp((v,I), F_profile['a'], method=interpol_kind,
                             bounds_error=False,fill_value=None)
            R_intp  = interp((v,I), F_profile['Nscattrate'], method=interpol_kind,
                             bounds_error=False,fill_value=None)
            def a(v,r): return a_intp(xi=(v,I_tot(r)))
            def R(v,r): return R_intp(xi=(v,I_tot(r)))
            
            if force_axis == '-v':
                v0_arr2         = v0_arr.copy()
                v0_arr2[:,0]   *= 0
                force_axis      = -v0_arr2/np.linalg.norm(v0_arr2,axis=-1)[:,None]
            elif isinstance(force_axis,(list,np.ndarray)):
                force_axis = np.atleast_2d(np.array(force_axis)/np.linalg.norm(force_axis)) +v0_arr*0
            else:
                raise Exception('input argument <force axis> has to be given!')
                
        else:
            position_dep = False
            from scipy.interpolate import interp1d
            a   = interp1d(v, F_profile['a'], kind=interpol_kind)
            R   = interp1d(v, F_profile['Nscattrate'], kind=interpol_kind)
            force_axis = np.array(force_axis)/np.linalg.norm(force_axis)
        
        # Differential equation calculation:
        #__________________________________
        def ode_MC1D(t,y,force_axis,position_dep):
            dy      = np.zeros(6+1)
            v_proj = np.sum(y[:3]*force_axis)
            if position_dep:
                dy[:3] = a(v_proj,y[3:6])*force_axis
                dy[-1] = R(v_proj,y[3:6])
            else:
                dy[:3] = a(v_proj)*force_axis
                dy[-1] = R(v_proj)
            dy[3:6] = y[:3]
            return dy
        #__________________________________
        
        iterator = tqdm(v0_arr,smoothing=0.0) if verbose else v0_arr
        for i,v0 in enumerate(iterator):
            sol = solve_ivp(ode_MC1D, (0.,t_int[i]), np.array([*v0, *r0_arr[i], 0.0]),
                            t_eval=t_eval, args=(force_axis[i],position_dep),
                            **kwargs)
            if save_scipy_sols:
                self.trajectory_results['sols'].append(sol)
            for key, indices in dict(photons=-1, r=[3,4,5], v=[0,1,2]).items():
                self.trajectory_results['final_values'][key][i] = sol.y[indices,-1]
        
    #%%
    def calc_OBEs(self, t_int=20e-6, t_start=0., dt=None, t_eval = [],
                  magn_remixing=False, freq_clip_TH=500, steadystate=False,
                  position_dep=False, rounded=False,
                  verbose=True, return_fun=return_fun_default,
                  **kwargs):
        """Calculate the time evolution of the single level populations with
        the optical Bloch equations.

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
            decide at which time points to calculate the solution. If dt='auto',
            an appropriate time step is chosen by using the smallest Rabi frequency
            between a single transition and single laser component.
            The default is None.
        t_eval : list or numpy array, optional
            If it desired to get the solution of the ode solver only at 
            specific time points, the `t_eval` argument can be used to specify 
            these points. If `_eval` is given, the `dt` argument is ignored.
            The default is [].
        magn_remixing : bool, optional
            if True, the magnetic field, which is defined in the instance
            :class:`~System.Bfield` contained in this class, is considered in the
            calculation. Otherwise, the magnetic field is set to zero.
            The default is False.
        freq_clip_TH : float or string, optional
            determines the threshold frequency at which the coupling of a single
            transition detuned by a frequency from a specific laser component
            is neglected. If a float is provided, only the transitions with
            detunings smaller than `freq_clip_TH` times Gamma[0] are driven by the
            light field. If `freq_clip_TH` == 'auto', the threshold frequencies for
            all transitions are chosen seperately by considering the transition
            strengths and intensities of each laser component.            
            The default is 500.
        steadystate : bool, optional
            determines whether the equations are propagated until a steady
            state or periodic quasi-steady state is reached. The dictionary
            ``steadystate`` of this class specifies the steady state conditions.
            The default is False.
        position_dep : bool, optional
            determines whether to take position of the particle in an Gaussian
            intensity distribution of the laser beams into account.
            The default is False.
        rounded : float, optional
            if specified, all frequencies and velocities are rounded to the frequency
            `rounded` in units of max(Gamma).
            The default is False.
        verbose : bool, optional
            whether to print additional information like execution time or the
            scattered photon number. The default is True.
        return_fun : function-type, optional
            if `mp` == True, the returned dictionary of this function determines
            the quantities which are save for every single parameter configuration.
            The default is None.
        **kwargs : keyword arguments, optional
            other options of the `solve_ivp` scipy function can be specified
            (see homepage of scipy for further information).

        Note
        ----
        function creates attributes 
        
        * ``N`` : solution of the time dependent populations N,
        * ``Nscatt`` : time dependent scattering number,
        * ``Nscattrate`` : time dependent scattering rate,
        * ``photons``: totally scattered photons
        * ``args``: input arguments of the call of this function
        * ``t`` : times at which the solution was calculated
        """
        self.calcmethod = 'OBEs'
        #___input arguments of this called function
        self.args       = locals()
        self.check_config(raise_Error=True)
        
        #___parameters belonging to the levels
        self.levels.calc_all()
        # for dimensionless time units
        freq_unit       = self.levels.calc_Gamma().max() # choose the linewidth of the first electronic state as unit
        self.freq_unit  = freq_unit
        #frequency differences between the ground and excited states (delta)
        self.om_eg      = self.levels.calc_freq()/freq_unit
        if rounded:
            self.om_eg  = np.around(self.om_eg/rounded)*rounded
        
        #___start multiprocessing if desired
        if self._identify_iter_params() or self.lasers._identify_iter_params()[0]:
            return self._start_mp()
        
        #___parameters belonging to the lasers (and partially to the levels)
        # Rabi frequency in dimensionless units (lNum,uNum,pNum)
        self.calc_Rabi_freqs(position_dep=position_dep)
        # wave vectors k (no unit vectors)
        self.k  = self.lasers.getarr('k')*self.lasers.getarr('kabs')[:,None]
        # laser frequencies omega_k
        if rounded:
            self.om_k   = np.around(self.lasers.getarr('omega')/freq_unit/rounded)*rounded \
                        - np.around(np.dot(self.k,self.v0)/freq_unit/rounded)*rounded
        else:
            self.om_k   = (self.lasers.getarr('omega') - np.dot(self.k,self.v0))/freq_unit
        
        #___magnetic remixing of the ground states and excited states
        if magn_remixing:
            betaB  = self.Bfield.Bvec_sphbasis/(hbar*freq_unit/self.Bfield.mu_B)
        else:
            betaB  = np.array([0.,0.,0.])
            
        #coefficients h to neglect highly-oscillating terms of the OBEs (with frequency threshold freq_clip_TH)
        self.om_gek = self.om_eg[:,:,None] - self.om_k[None,None,:]
        if freq_clip_TH == 'auto':
            FWHM = np.sqrt(self.levels.calc_Gamma()[None,:,None]**2 + 2*np.abs(self.Rabi_freqs)**2)/freq_unit #in dimensionless units
            self.h_gek  = np.where(np.abs(self.om_gek) < 8*FWHM/2, 1.0, 0.0)
            self.h_gege = np.where(np.abs(self.om_eg[:,:,None,None]-self.om_eg[None,None,:,:])\
                                   < 8*np.max(FWHM)/2, 1.0, 0.0)
        else:
            self.h_gek  = np.where(np.abs(self.om_gek) < freq_clip_TH, 1.0, 0.0)
            self.h_gege = np.where(np.abs(self.om_eg[:,:,None,None]-self.om_eg[None,None,:,:])\
                                   < freq_clip_TH, 1.0, 0.0)
        
        #___coefficients for new defined differential equations
        self.Gfd = 1j/2*np.exp(1j*self.lasers.getarr('phi')[None,None,:])*self.h_gek*self.Rabi_freqs/freq_unit
        self.betamu = tuple(1j* np.dot(self.levels.calc_muMat()[i], np.flip(betaB*np.array([-1,1,-1])))
                            for i in range(2) )
        self.dd = self.h_gege*(self.levels.calc_dMat()[:,:,None,None,:]\
                               *self.levels.calc_dMat()[None,None,:,:,:]).sum(axis=-1)
        self.ck_indices = (tuple(np.where(self.Gfd[i,:,:] != 0.0) for i in range(self.levels.lNum)),
                           tuple(np.where(self.Gfd[:,i,:] != 0.0) for i in range(self.levels.uNum)))
        self.ck_indices = (tuple( np.array([i[0],i[1]]) for i in self.ck_indices[0] ),
                           tuple( np.array([i[0],i[1]]) for i in self.ck_indices[1] ))
        
        #___specify the initial (normalized) occupations of the levels
        #   and transform these values into the density matrix elements N0mat
        N0mat = self.initialize_N0(return_densitymatrix=True)
        
        if verbose: print('Solving ode with OBEs...', end='')
        start_time = time.perf_counter()
        
        #___if steady state is wanted, multiple calculation steps of the OBEs
        #___have to be performed while the occupations between this steps are compared
        if not steadystate:
            self._evaluate(t_start, t_int, dt, N0mat)
            self.step = 0
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
                t_int = 2*pi/(freq_unit*self.args['rounded'])
            elif self.steadystate['period'] == 'standingwave':
                if np.linalg.norm(self.v0) != 0: #if v0==0, then t_int is not changed and thus used for int time.
                    lambda_mean = (c/(self.om_eg*freq_unit/2/pi)).mean()
                    if np.any(np.abs(c/(self.om_eg*freq_unit/2/pi) /lambda_mean -1)>0.1e-2 ):#percental deviation from mean
                        print('WARNING: averaging over standing wave periods might not be accurate since the wavelengths differ.')
                    period = lambda_mean/np.linalg.norm(self.v0)#/2
                    t_int = period*(t_int//period+1) # int(t_int - t_int % period)
            self._evaluate(t_start, t_int, dt, N0mat)
            t_start = self.t[-1]
            N0mat   = self.ymat[:,:,-1]
            m1      = self.N.mean(axis=1)
            # if self.steadystate['period'] == None: t_int *= 0.1
            con1, con2 = self.steadystate['condition']
            for self.step in range(1,self.steadystate['maxiters']):
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
            if verbose: print(' calculation steps: ',self.step+1)
            
        #: execution time for the ODE solving
        self.exectime = time.perf_counter() - start_time
        self._verify_calculation()
        if return_fun == True: return {'system':self}
        elif return_fun: return return_fun(self)
    
    def _verify_calculation(self):
        dev_TH  = {'rateeqs':1e-8,'OBEs':1e-6}[self.calcmethod]
        dev     = abs(self.N[:,-1].sum() -1)
        #: Variable success indicates of the calcualtion could be verified.
        self.success = True
        self.message = ""
        if dev > dev_TH:
            message = 'Sum of populations not stable! Deviation: {:.2e}.\n'.format(dev)
            print('WARNING:', message)
            self.message += message
            self.success = False
        if np.any(self.N < -1e-3):
            message = 'Populations got negative!'
            print('WARNING:', message)
            self.message += message
            self.success = False
        
        # printing some information...    
        if self.args['verbose']:
            print("Execution time: {:2.4f} seconds".format(self.exectime))
            for i,Ex_label in enumerate(self.levels.exstates_labels):
                print("Scattered Photons ({}): {:.6f}".format(Ex_label,self.photons[i]))
    
    def _start_mp(self):
        self.results = tools.multiproc(obj=deepcopy(self),kwargs=self.args)
        if self.multiprocessing['savetofile']:# multiprocessing.cpu_count() > 16:
            save_object(self)
            try:
                sys.path.append('../')
                import subprocess, sending_email
                hostname = subprocess.check_output('hostname').decode("utf-8")
                sending_email.send_message('Calculation complete!','File {} at Server {}'.format(self.description,hostname))
            except:
                pass
        return None
    
    def _identify_iter_params(self):
        """Identify which parameters are iterative arrays  to loop through
        including magnetic field strength and direction, initial position and
        velocity. This function is inevitable to determine whether multiple
        evaluations of the OBEs and rate equations are efficiently conducted 
        using the multiprocessing package from python (see :meth:`tools.multiproc`
        and :meth:`Lasersystem._identify_iter_params`).

        Returns
        -------
        iters_dict : dict
            Dictionary with all iterative parameters and their number of iterations.
        """
        iters_dict = {}
        
        for obj, ndim, label in [
                (self.Bfield.strength, 0, 'strength'),
                (self.Bfield.direction, 1, 'direction'),
                (self.r0, 1, 'r0'),
                (self.v0, 1, 'v0'),
                ]:
            if np.array(obj).ndim != ndim:
                iters_dict[label] = len(obj)        
        
        return iters_dict
    
    def reset_N0(self):
        """Reset (last created) initial population self.N0 and initial populations
        for all electronic states"""
        self.N0     = np.array([]) #: initial population of all levels
        for ElSt in self.levels.electronic_states:
            ElSt.N0 = []
    
    def initialize_N0(self,return_densitymatrix=False,random=False):
        """
        Initialize initial populations as a starting point for the OBEs or
        rate equations using pre-defined population attribute ``N0`` of the
        electronic states :class:`~MoleCool.Levelsystem.ElectronicState`.

        Parameters
        ----------
        return_densitymatrix : bool, optional
            Whether the populations should be transformed into density matrix
            or one-dimensional population array. The default is False.
        random : bool, optional
            Whether the popultions are sampled from a random distirbution.
            The default is False.

        Returns
        -------
        N0mat : numpy.ndarray
            density matrix with populations on the diagonal.
        """
        #___specify the initial (normalized) occupations of the levels
        N,iNum = self.levels.N, self.levels.iNum
        if random:
            self.N0 = np.random.rand(N)
        else:
            if np.any([len(ElSt.N0) for ElSt in self.levels.electronic_states]):
                self.N0 = np.zeros(N)
                for ElSt in self.levels.electronic_states:
                    if len(ElSt.N0) != 0:
                        i_ElSt = self.levels.index_ElSt(ElSt,include_Ngrs_for_exs=True)
                        self.N0[i_ElSt:(i_ElSt+ElSt.N)] = ElSt.N0
            else:
                self.N0 = np.array(self.N0, dtype=float)
                if len(self.N0) == 0:
                    if 'v' in self.levels.grstates[0][0].QuNrs:
                        N0_indices = [i for i,st in enumerate(self.levels.grstates[0]) if st.v==0] 
                        if len(N0_indices) == 0:
                            self.N0     = np.ones(N)
                        else:
                            self.N0      = np.zeros(N)
                            for i in N0_indices:
                                self.N0[i] = 1.0
                    else:
                        self.N0     = np.ones(N)
                else:
                    if len(self.N0) != N:
                        if len(self.N0) == N+iNum:
                            self.N0 = self.N0[:N]
                        else:
                            raise ValueError('Wrong size of N0')
        self.N0 /= self.N0.sum() #initial populations are always normalized
        
        if iNum > 0:
            self.N0 = np.array([*self.N0,*self.N0[self.levels.lNum-iNum:self.levels.lNum]])
        
        if return_densitymatrix:
            N = N +iNum
            N0mat = np.zeros((N,N),dtype=np.complex64)
            if random:
                print('MUST be modified that the density matrix elements that are twice apparent are equal')
                N0mat = (np.random.rand(N,N) + 1j*np.random.rand(N,N))/2
                N0mat +=  np.conj(N0mat).T 
            #transform these initial values into the density matrix elements N0mat
            N0mat[(np.arange(N),np.arange(N))] = self.N0
            return N0mat

    def get_N(self, return_sum=True, **QuNrs):
        """
        Retrieve the time-dependent populations as results after calculating
        the dynamics. Can be used to either obtain the populations of all
        individual levels or to conveniently combine the populations for only a 
        subset of levels with specific Quantum numbers.

        Parameters
        ----------
        return_sum : bool, optional
            Whether to sum up of all levels. The default is True.
        **QuNrs : kwargs
            Keyword arguments as Quantum numbers can be provided for only a subset
            of levels with specific Quantum numbers, e.g. v=0. If empty, all
            levels are considered. If only a specific ground or excited
            electronic state should be included, add e.g. `gs='X'` or `exs='A'`
            as a first QuNr keyword.

        Returns
        -------
        np.ndarrays
            Time-dependent population(s).
        """
        if not QuNrs:
            return self.N
        
        states = self.levels.states
        inds = np.array([i for i,st in enumerate(states) if st.check_QuNrvals(**QuNrs)])
        if return_sum:
            return self.N[inds,:].sum(axis=0)
        else:
            return self.N[inds,:]    

    def get_Nscattrate(self,sum_over_ElSts=False):
        """Calculate time dependent scattering rate.

        Parameters
        ----------
        sum_over_ElSts : bool, optional
            Whether to sum over multiple electronic excited states.
            The default is False.

        Returns
        -------
        numpy.ndarray
            Time dependent scattering rate.
        """
        Nscattrate_arr = self.levels.calc_Gamma()[:,None]*self.N[self.levels.lNum:,:]
        if not sum_over_ElSts:# and (len(self.levels.exstates_labels) > 1)
            Nscattrate_summed = np.zeros((len(self.levels.exstates_labels),self.N.shape[1]))
            N_states = 0
            for i,ElSt in enumerate(self.levels.exstates):
                N = ElSt.N
                Nscattrate_summed[i,:] = np.sum(Nscattrate_arr[N_states:N_states+N,:],axis=0)
                N_states += N
            return Nscattrate_summed
        else:
            return np.sum(Nscattrate_arr,axis=0)
        
    def get_Nscatt(self,sum_over_ElSts=False):
        """Calculate time dependent scatterd photon number.

        Parameters
        ----------
        sum_over_ElSts : bool, optional
            Whether to sum over multiple electronic excited states.
            The default is False.

        Returns
        -------
        numpy.ndarray
            Time dependent scattered photon number.
        """        
        return cumulative_trapezoid(self.get_Nscattrate(sum_over_ElSts=sum_over_ElSts), 
                        self.t, initial = 0.0, axis=-1)
    
    def get_photons(self,sum_over_ElSts=False):
        """Calculate totally scattered photon number.

        Parameters
        ----------
        sum_over_ElSts : bool, optional
            Whether to sum over multiple electronic excited states.
            The default is False.

        Returns
        -------
        numpy.ndarray
            Totally scattered photon number.
        """        
        return np.transpose(self.get_Nscatt(sum_over_ElSts=sum_over_ElSts))[-1]
    
    #___compute several physical variables using the solution of the ODE
    #: time dependent scattering rate
    Nscattrate  = property(get_Nscattrate)
    #: time dependent scattering number
    Nscatt      = property(get_Nscatt)
    #: totally scattered photons
    photons     = property(get_photons)
                
    def _evaluate(self,t_start,t_int,dt,N0mat): #for OBEs only?!
        freq_unit = self.freq_unit
        #___determine the time points at which the ODE solver should evaluate the equations    
        if len(self.args['t_eval']) != 0:
            self.t_eval = np.array(self.args['t_eval'])
        else:
            if dt == 'auto': #must be tested!?!
                dt = 1/np.max(self.h_gek*np.sqrt(self.om_gek**2*freq_unit**2+np.abs(self.Rabi_freqs)**2)/2/pi)/8.11 #1/8 of one Rabi-oscillation
            if dt != None and dt < t_int:
                self.t_eval = np.linspace(t_start,t_start+t_int,int(t_int/dt)+1)
            else:
                self.t_eval, T_eval = None, None
        if np.all(self.t_eval) != None:
            T_eval = self.t_eval * freq_unit
        
        #___transform the initial density matrix N0mat in a vector for the ode
        self.y0 = self._density_mat2vec(N0mat)
        
        # ---------------Ordinary Differential Equation solver----------------
        # solve initial value problem of the ordinary first order differential equation with scipy
        lNum,uNum,iNum,pNum = self.levels.lNum,self.levels.uNum,self.levels.iNum,self.lasers.pNum
        kwargs = self.args['kwargs']
        # sol = solve_ivp(ode0_OBEs, (t_start*freq_unit,(t_start+t_int)*freq_unit),
        #                 self.y0, t_eval=T_eval, **kwargs,
        #                 args=(lNum,uNum,pNum,self.G,self.f,self.om_eg,self.om_k,
        #                       betaB,self.dMat,self.muMat,
        #                       self.M_indices,h_gek,h_gege,self.phi)) # delete?
        # sol = solve_ivp(ode1_OBEs, (t_start*freq_unit,(t_start+t_int)*freq_unit),
        #                 self.y0, t_eval=T_eval, **kwargs,
        #                 args=(lNum,uNum,pNum, self.M_indices,
        #                       self.Gfd,self.om_gek,self.betamu,self.dd))
        # sol = solve_ivp(ode1_OBEs_opt2, (t_start*freq_unit,(t_start+t_int)*freq_unit), #<- optimized form for one el. ex. state!
        #                 self.y0, t_eval=T_eval, **kwargs,
        #                 args=(lNum,uNum,pNum, self.M_indices,
        #                       self.Gfd,self.om_gek,self.betamu,self.dd,self.ck_indices))
        if iNum == 0:
            sol = solve_ivp(ODEs.ode1_OBEs_opt3, (t_start*freq_unit,(t_start+t_int)*freq_unit),  #<- can also handle two electr. states with diff. Gamma
                            self.y0, t_eval=T_eval, **kwargs,
                            args=(lNum,uNum,pNum, self.levels.calc_M_indices(),
                                  self.Gfd,self.om_gek,self.betamu,self.dd,self.ck_indices,
                                  self.levels.calc_Gamma()/freq_unit))
        else:
            sol = solve_ivp(ODEs.ode1_OBEs_opt4, (t_start*freq_unit,(t_start+t_int)*freq_unit),  #<- can also handle two electr. states with diff. Gamma
                            self.y0, t_eval=T_eval, **kwargs,
                            args=(lNum,uNum,iNum,pNum, self.levels.calc_M_indices(),
                                  self.Gfd,self.om_gek,self.betamu,self.dd,self.ck_indices,
                                  self.levels.calc_Gamma()/freq_unit))
        self.sol = sol
        self.ymat = self._density_vec2mat(sol.y)
        #: solution of the time dependent populations N
        self.N = np.real(self.ymat[(np.arange(self.levels.N),np.arange(self.levels.N))])
        #: array of the times at which the solutions are calculated
        self.t = sol.t/freq_unit
    
    def _density_mat2vec(self,mat):
        """Transform the time dependent density matrix from matrix form to
        solution vector of the ODEs. Inverse operation to _density_vec2mat.
        """
        if mat.ndim != 2:
            raise Exception('Matrix must be 2-dimensional')
        if mat.shape[0] != mat.shape[1]:
            raise Exception('Matrix must be square matrix')
        N       = mat.shape[0]
        vec  = np.zeros( N*(N+1) )
        count   = 0
        for i in range(N):
            for j in range(i,N):
                vec[count]   = mat[i,j].real
                vec[count+1] = mat[i,j].imag
                count += 2
        return vec
    
    def _density_vec2mat(self,vec):
        """Transform the solution vector of the time dependent density matrix
        into matrix form. Inverse operation to _density_mat2vec.
        """
        N,iNum  = self.levels.N, self.levels.iNum
        if vec.shape[0] != (N+iNum)*(N+iNum+1):
            raise Exception('Shape[0] of vector must have the length N+iNum')
        mat    = np.zeros((N+iNum,N+iNum,vec.shape[-1]),dtype=np.complex64)
        count   = 0
        for i in range(N+iNum):
            for j in range(i,N+iNum):
                mat[i,j,:] = vec[count] + 1j* vec[count+1]
                count += 2
        mat    = mat[:N,:N]
        if np.any(np.abs(mat[(np.arange(N),np.arange(N))].imag) > 1e-13):
            warnings.warn('Populations got an imaginary part > 1e-13')
        mat    += np.conj(np.transpose(mat,axes=(1,0,2))) #is diagonal remaining purely real or complex?
        mat[(np.arange(N),np.arange(N))] *=0.5
        return mat
    
    def check_config(self,raise_Error=False):
        """Check System configuration for simulating internal dynamics."""
        if self.calcmethod == 'rateeqs':
            #pre-defined kwargs for solve_ivp function
            kwargs_default = {'method':'LSODA', 'max_step':10e-6} 
            self.args['kwargs'] = dict(kwargs_default,**self.args['kwargs'])
        self.lasers.check_config(raise_Error=raise_Error)
        self.levels.check_config(raise_Error=raise_Error)
        if 'trajectory' in self.args:
            if (self.args['trajectory']) and (self.levels.mass==0.0):
                raise ValueError('No mass is provided for trajectory to be calculated')
        # check if some lasers are completely off to some states or if some states are not addressed by any laser!?
            
    def __set_v0r0(self, vec=[0,0,0], direction=[], label='v0'):
        vec = np.array(vec, dtype=np.float64)
        
        if isinstance(direction, str):
            direction = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],'':[]}[direction]
        if np.any(direction):
            direction = np.array(direction)/np.linalg.norm(direction)
            if direction.shape != (3,):
                raise Exception(f"direction vector must be of shape (3,) instead of {direction.shape}")
            vec = direction[None,:]*vec[:,None]
        
        if vec.ndim not in [1,2]:
            raise Exception(f"{label} must have one or two dimensions instead of {vec.ndim}")
        if vec.shape[-1] != 3:
            raise Exception(f"Last dimension of {label} must be of length 3 instead of {vec.shape[-1]}")
        
        return vec
    
    def set_v0(self, v0=[0,0,0], direction=[]):
        """Set initial velocity of the particle(s).

        Parameters
        ----------
        v0 : list or np.ndarray, optional
            Velocity array. Can be either a one-dimensional array with three
            entries for the x, y and z component, a two dimensional array
            where the first dimension corresponds to different particles' 
            velocities, or a one dimensional array with absolute velocity
            values while the argument ``direction`` defines the direction.
            The default is [0,0,0].
        direction : str, list or np.ndarray, optional
            Direction of the absolute velocity values defined as the argument
            ``v0``. Can be either a str (x,y,z) or list / array of length 3.
            The default is [].

        Returns
        -------
        np.ndarray
            velocity array (1 or 2-dimensional with the shape (number of
            particles, 3) in cartesian coordinates.
        """
        self.__v0 = self.__set_v0r0(vec=v0, direction=direction, label='v0')
        return self.__v0
    
    def set_r0(self, r0=[0,0,0], direction=[]):
        """Set initial position of the particle(s).

        Parameters
        ----------
        r0 : list or numpy.ndarray, optional
            Position array. Can be either a one-dimensional array with three
            entries for the x, y and z component, a two dimensional array
            where the first dimension corresponds to different particles' 
            positions, or a one dimensional array with absolute position
            values while the argument ``direction`` defines the direction.
            The default is [0,0,0].
        direction : str, list or numpy.ndarray, optional
            Direction of the absolute position values defined as the argument
            ``r0``. Can be either a str (x,y,z) or list / array of length 3.
            The default is [].

        Returns
        -------
        numpy.ndarray
            position array (1 or 2-dimensional with the shape (number of
            particles, 3) in cartesian coordinates.
        """
        self.__r0 = self.__set_v0r0(vec=r0, direction=direction, label='r0')
        return self.__r0
    
    @property
    def v0(self):
        """Initial velocity vector or array of vectors."""
        if 'v0' in self.__dict__:
            val = self.__dict__['v0']
        else:
            val = self.__v0
        return val
    
    @v0.setter
    def v0(self,val):
        self.set_v0(v0=val)
        
    @property
    def r0(self):
        """Initial position vector or array of vectors."""
        if 'r0' in self.__dict__:
            val = self.__dict__['r0']
        else:
            val = self.__r0
        return val
    
    @r0.setter
    def r0(self,val):
        self.set_r0(r0=val)
    
    def draw_levels(self,GrSts=None,ExSts=None,branratios=True,lasers=True,
                    QuNrs_sep=[], br_fun='identity', br_TH=0.01, # 1e-16 default
                    freq_clip_TH='auto', cmap='viridis',yaxis_unit='MHz'):
        """Draw all levels of certain Electronic states sorted
        by certain Qunatum numbers. Additionally, the branching ratios and the
        transitions addressed by the lasers can be added.

        Parameters
        ----------
        GrSts : list of str, optional
            list of labels of ground electronic states to be displayed.
            The default is None which corresponds to all ground states.
        ExSts : list of str, optional
            list of labels of excited electronic states to be displayed.
            The default is None which corresponds to all excited states.
        branratios : bool, optional
            Whether to show the branching ratios. The default is True.
        lasers : bool, optional
            Whether to show the transitions addressed by the lasers.
            By default it is set to True when lasers are defined.
        QuNrs_sep : list of str or tuple of two lists of str, optional
            Quantum numbers for separating all levels into subplots.
            For example the levels can be grouped into subplots by the vibrational
            Quantum number, i.e. ['v'] or (['v'],['v']) for ex. and gr. states.
        br_fun : str or callable, optional
            Function to be applied onto the branching ratios. Can be either
            'identity', 'log10', 'sqrt', or a custom defined function.
            The default is 'identity'.
        br_TH : float, optional
            Threshold for the branching ratios to be shown. The default is 0.01.
        freq_clip_TH : TYPE, optional
            Same argument as in OBE's calculation method ':func:`calc_OBEs`.
            The default is 'auto'.
        cmap : str, optional
            Colormap to be applied to the branching ratios. The default is 'viridis'.
        yaxis_unit : str or float, optional
            Unit of the y-axis. Can be either 'MHz','1/cm', or 'Gamma' for the
            natural linewidth. Alternatively, an arbitrary unit (in MHz) can be
            given as float. Default is 'MHz'.
        """
        # from mpl.patches import ConnectionPatch
        def draw_line(subfig,axes,xys,arrowstyle='->',dxyB=[0.,0.],**kwargs):
            con = mpl.patches.ConnectionPatch(
                xyA=xys[0], coordsA=axes[0].transData,
                xyB=xys[1]+dxyB, coordsB=axes[1].transData,
                arrowstyle=arrowstyle,
                shrinkB=1,
                **kwargs
                )
            subfig.add_artist(con)
            axes[1].plot(*(xys[1]+dxyB)) #invisible line for automatically adjusting axis limits
            
        levels = self.levels
        if GrSts == None:
            GrSts = levels.grstates
        else:
            GrSts = [levels.__dict__[label] for label in GrSts]
        if ExSts == None:
            ExSts = levels.exstates
        else:
            ExSts = [levels.__dict__[label] for label in ExSts]
            
        # create big figure, and nested subfigures and subplot axes
        fig          = plt.figure(constrained_layout=True)
        # subfigures subfigs for dividing main axes content and axes for legends
        subfigs      = fig.subfigures(1, 2, hspace=0.0, width_ratios=[5,1.5])
        # Two main subfigures for ground and excited electronic state
        height_ratios_main = [1, 2]
        subfigs_main = subfigs[0].subfigures(2, 1, wspace=0.0, height_ratios=height_ratios_main )
        subfigs_ExSts= subfigs_main[0].subfigures(1, len(ExSts), hspace=0.0, width_ratios=[Exs.N for Exs in ExSts],squeeze=False )[0]
        subfigs_GrSts= subfigs_main[1].subfigures(1, len(GrSts), hspace=0.0, width_ratios=[Grs.N for Grs in GrSts],squeeze=False )[0]
        # Two subfigure instances for legend and colorbar with each a single subplot axis
        subfigs_leg  = subfigs[1].subfigures(2, 1, wspace=0.0, height_ratios=height_ratios_main )
        axs_legend = subfigs_leg[1].subplots(1, 1)
        ax_cbar   = subfigs_leg[0].subplots(1, 1)
        tools.make_axes_invisible(axs_legend)
        #needed for drawing arrows on top over multiple figures
        subfig_last = subfigs_GrSts[-1]
        
        # draw only levels first
        if isinstance(QuNrs_sep,tuple): QuNrs_sep_u, QuNrs_sep_l = QuNrs_sep
        else:                           QuNrs_sep_u, QuNrs_sep_l = 2*[QuNrs_sep]
        
        coords_u = [ElSt.draw_levels(fig=subfigs_ExSts[i], QuNrs_sep=QuNrs_sep_u,
                                     yaxis_unit=yaxis_unit, ylabel=not bool(i),
                                     xlabel_pos='top')
                                     for i, ElSt in enumerate(ExSts)]
        coords_l = [ElSt.draw_levels(fig=subfigs_GrSts[i], QuNrs_sep=QuNrs_sep_l,
                                     yaxis_unit=yaxis_unit)
                                     for i, ElSt in enumerate(GrSts)]
        
        try:
            get_cmap = mpl.cm.get_cmap
        except AttributeError:
            get_cmap = mpl.colormaps.get_cmap
        cmap        = get_cmap(cmap)
        
        # map branching ratios onto a color using a certain function:
        if branratios:
            branratios  = levels.calc_branratios()
            from scipy.interpolate import interp1d
            if not callable(br_fun):
                br_fun_dict = {'log10':np.log10,'identity':lambda x:x,'sqrt':np.sqrt}
                if not br_fun in br_fun_dict: raise Exception('Not valid argument given for br_fun!')
                br_fun      = br_fun_dict[br_fun]
            # isolate all branratios over a certain threshold in a separate flattened array and apply function
            br_flat     = br_fun(branratios[branratios > br_TH])
            # map branratio onto interval [0,1] for colormap
            map_br      = interp1d([br_flat.min(),br_flat.max()],[1,0])
            
            # iterate over states and draw branratios
            for GrSt,coords_l_ in zip(GrSts,coords_l):
                for ExSt,coords_u_ in zip(ExSts,coords_u):
                    if GrSt == ExSt: continue
                    Ng = self.levels.index_ElSt(GrSt,gs_exs='gs')
                    Ne = self.levels.index_ElSt(ExSt,gs_exs='exs')
                    for l,u in np.argwhere(branratios[Ng:Ng+GrSt.N,Ne:Ne+ExSt.N] > br_TH): #does not work when ExSts are switched, e.g. ['B','A']??!
                        draw_line(subfig_last,
                                  axes=(coords_l_['axes'][l], coords_u_['axes'][u]),
                                  xys=(coords_l_['xy'][l], coords_u_['xy'][u]),
                                  arrowstyle='-',dxyB=[-0.2,0],alpha=0.5,
                                  linewidth=0.6,linestyle='--',
                                  color= cmap(map_br(br_fun(branratios[Ng+l,Ne+u]))))
            # draw colorbar
            norm        = mpl.colors.Normalize(vmax=br_flat.max(),vmin=br_flat.min())
            bounds      = np.array(ax_cbar.get_position().bounds)
            ax_cbar.set_position(bounds*np.array([1,1,0.2,1])) #left bottom width height
            ax_cbar.tick_params(labelsize='small')
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap.reversed()),
                         ticks=np.linspace(br_flat.min(),br_flat.max(),5),
                         format='{:.2%}'.format,
                         cax=ax_cbar, orientation='vertical', label='bran. ratio')
        else:
            tools.make_axes_invisible(ax_cbar)
            
        # draw lasers onto their addressing transitions as arrows:
        if self.lasers.pNum == 0: lasers = False
        if lasers:
            ls_fun = lambda x: ['-','-.',':'][x//10]
            self.calc_OBEs(t_int=1e-11, t_start=0., dt=None, t_eval = [],
                          magn_remixing=False, freq_clip_TH=freq_clip_TH, steadystate=False,
                          position_dep=False, rounded=False,
                          verbose=False, return_fun=None, method='RK45')
            
            for GrSt,coords_l_ in zip(GrSts,coords_l):
                for ExSt,coords_u_ in zip(ExSts,coords_u):
                    if GrSt == ExSt: continue
                    Ng = self.levels.index_ElSt(GrSt,gs_exs='gs')
                    Ne = self.levels.index_ElSt(ExSt,gs_exs='exs')
                    for l,u,k in np.argwhere(np.abs(self.Gfd[Ng:Ng+GrSt.N,Ne:Ne+ExSt.N,:])>0):
                        det = self.om_gek[Ng+l,Ne+u,k]*self.freq_unit/2/pi*1e-6/coords_u_['yaxis_unit']
                        draw_line(subfig_last,
                                  axes=(coords_l_['axes'][l], coords_u_['axes'][u]),
                                  xys =(coords_l_['xy'][l],   coords_u_['xy'][u]),
                                  arrowstyle='->',linestyle=ls_fun(k),dxyB=[+0.2,det],
                                  alpha=0.8,color='C'+str(k))
            
            # legend for lasers
            import matplotlib.lines as mlines
            handles = [mlines.Line2D([],[],lw=1,color='C'+str(k),ls=ls_fun(k),
                                     label='{:d}: {:.0f}, {:.1f}'.format(
                                         k, la.lamb*1e9, la.P*1e3)
                                     )
                for k,la in enumerate(self.lasers)]
                
            legend = axs_legend.legend(handles=handles,loc='upper left',
                                       bbox_to_anchor=(0, 1),fontsize='x-small',
                                       title='i: $\lambda$ [nm], $P$ [mW]')
            plt.setp(legend.get_title(),fontsize='x-small')

#%%
if __name__ == '__main__':
    system = System(description='test_System_mod',load_constants='138BaF')
    
    # Build level scheme
    system.levels.add_all_levels(v_max=0)
    system.levels.X.del_lossstate()
    
    # Add laser configuration
    system.lasers.add_sidebands(
        lamb        = 859.83e-9,
        P           = 20e-3,
        offset_freq = 19e6,
        mod_freq    = 39.33e6,
        sidebands   = [-2, -1, 1, 2],
        ratios      = [0.8, 1, 1, 0.8]
    )
    
    # Magnetic field
    system.Bfield.turnon(strength=5e-4, direction=[1, 1, 1])
    
    # Dynamics simulations
    system.calc_OBEs(t_int=8e-6, dt=1e-9,
                     magn_remixing=True)
    system.calc_rateeqs(t_int=8e-6, magn_remixing=True,
                        position_dep=True)
    system.plot_N()
    