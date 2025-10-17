# -*- coding: utf-8 -*-
"""
This module contains all classes and functions to define a System including
multiple :class:`Laser` objects.

Example
-------
Below an empty Lasersystem is created and a single Laser with wavelength 860nm
and Power 20 mW with linear polarization is added::
    
    from MoleCool import Lasersystem
    lasers = Lasersystem()
    lasers.add(860e-9,20e-3,'lin')

Tip
---
Every object of the classes :class:`~MoleCool.Lasersystem.Lasersystem` or
:class:`~MoleCool.Lasersystem.Laser` class can be printed to display
all attributes via::
    
    print(lasers)
    print(lasers[0])
    
To delete all instances use this command::
    
    del lasers[:]
"""
import numpy as np
import pandas as pd
from scipy.constants import c,h,hbar,pi,g
from numba import jit
import matplotlib.pyplot as plt
import warnings
from MoleCool import tools
#%%
class Lasersystem:
    def __init__(self,freq_pol_switch=5e6):
        """System consisting of :class:`~MoleCool.Lasersystem.Laser` objects
        and methods to add them properly.
        These respective objects can be retrieved and also deleted by using the
        normal item indexing of a :class:`~MoleCool.Lasersystem.Lasersystem`
        object.
        
        Example
        -------
        ::
            
            from MoleCool import Lasersystem
            lasers = Lasersystem()
            lasers.add(lamb=860e-9,P=20e-3,pol='lin')
            lasers.add(lamb=890e-9,I=1000,FWHM=2e-3)            
            laser1 = lasers[0] # call first Laser object included in lasers
            del lasers[-1] # delete last added Laser object
            
            print(lasers[0])
            print(lasers)

        Parameters
        ----------
        freq_pol_switch : float, optional
            Specifies the frequency (without 2pi) with which the polarization is
            switched if the polarization switching is enabled. The default is 5e6.
        """
        self.entries = []
        #: float: Polarization switching frequency. Default is 5e6.
        self.freq_pol_switch = freq_pol_switch 
        self.intensity_func = None
        self.intensity_func_sum = None
        self.pd_display_options = [
            'display.float_format',     '{:.3e}'.format,
            "display.max_rows",         None,
            "display.max_columns",      None,
            ]

    def add(self,lamb=860e-9,P=20e-3,pol='lin',**kwargs):
        """Add a :class:`~MoleCool.Lasersystem.Laser` instance to the laser
        system.
        
        Note
        ----
        This is the same as::
            
            from MoleCool import Laser
            lasers.entries.append(Laser())
        
        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments. Same as in the ``__init__`` method of
            the class :class:`~MoleCool.Lasersystem.Laser`.
        """
        self.entries.append( Laser( lamb=lamb, P=P, pol=pol, **kwargs) ) 
        self.intensity_func = None
        self.intensity_func_sum = None
    
    def getarr(self, attr):
        """Get an array with a specific attribute of all included
        :class:`Laser` objects.

        Parameters
        ----------
        attr : str
            Laser attribute, e.g. ``lamb`` or ``P``.

        Returns
        -------
        numpy.ndarray
            Array with the lasers' attributes.
        """
        self.check_config()
        if not attr in dir(self[0]):
            raise ValueError('The attribute {} is not included in the Laser objects'.format(attr))
        if attr == 'f_q': 
            dtype = complex
        else:
            dtype = float
        return np.array([getattr(la,attr) for la in self],dtype=dtype)
    
    def _identify_iter_params(self):
        """Identify which parameters are iterative arrays  to loop through.
        This function is inevitable to determine whether multiple
        evaluations of the OBEs and rate equations are efficiently conducted 
        using the multiprocessing package from python
        (see :meth:`tools.multiproc`).

        Returns
        -------
        iters_dict : dict
            Dictionary with all iterative parameters and their number of iterations.
        """
        # loop through laser objects to get to know which variables have to get
        # iterated and how many iterations
        #--> for the dictionaries used here it'S important that the order is ensured
        #    (this is the case since python 3.6 - now (3.8))
        laser_list = []
        laser_iters_N = {}
        for l1,la in enumerate(self):
            laser_dict = {}
            for key in ['omega','freq_Rabi','I','phi','beta','k','r_k','f_q']:
                value = la.__dict__[key]
                if (np.array(value).ndim == 1 and key not in ['k','r_k','f_q']) \
                    or (np.array(value).ndim == 2 and key in ['k','r_k','f_q']):
                    laser_dict[key] = value
                    laser_iters_N[key] = len(value)
            laser_list.append(laser_dict)
        
        return laser_iters_N, laser_list
        
    def add_sidebands(self,lamb=860e-9,offset_freq=0.0,mod_freq=1e6,
                      ratios=None,sidebands=[-1,1],**kwargs):
        """Add multiple :class:`~MoleCool.Lasersystem.Laser` instances as
        sidebands e.g. to drive multiple hyperfine transitions.
        The individual sidebands are detuned from the center
        frequency by the modulation frequency ``mod_freq`` times the values
        in the list ``sidebands``, i.e. for ``mod_freq=1e6`` and
        ``sidebands=[-1,0,2]``, the sidebands are detuned by -1 MHz, 0 MHz
        and 2 MHz.
        The center frequency is given by the wavelength lamb and an additional
        general offset frequency ``offset_freq``.
        
        Parameters
        ----------
        lamb : float
            wavelength of the main transition.
        P : float
            Power, i.e. sum of the powers of all sidebands.
            Alternativley the sum of the intensities can be provided.
        I : float
            Sum of all sideband intensities. Can be provided instead of power P.
        offset_freq : float
            All Laser sidebands are all additionally detuned by the value of
            offset_freq (in Hz without 2 pi). Experimentally, this shift is often
            realized with an AOM. The default is 0.0.
        mod_freq : float
            starting from the offset-shifted center frequency, sideband Laserobjects
            are added with the detunings `sidebands`*`mod_freq` (without 2 pi).
        ratios : array_like, optional
            Power/ intensity ratios of the individual sidebands. Must be provided in the
            same order as the `mod_freq` parameter.
            (Will be normed to specify the individual sideband powers).
            The default is equally distributed power.
        sidebands : array_like, optional
            determines the number of sidebands and their detuning in units of
            the `mod_freq` parameter.
        **kwargs
            optional arguments  (see :class:`~MoleCool.Lasersystem.Laser`).
        """
        # compatibility for old parameter names:
        if 'AOM_shift' in kwargs:
            offset_freq = kwargs['AOM_shift']
            del kwargs['AOM_shift']
        if 'EOM_freq' in kwargs:
            mod_freq    = kwargs['EOM_freq']
            del kwargs['EOM_freq']
            
        # set equally distributed power ratios of not provided
        if np.all(ratios) == None:
            ratios = len(sidebands)*[1]
            
        if 'I' in kwargs:
            PorI     = 'I'
        elif 'P' in kwargs:
            PorI     = 'P'
        else:
            PorI     = 'P'
            kwargs['P'] = 20e-3
            
        mod_freqs = (np.array(sidebands)*np.expand_dims(mod_freq,axis=-1)).T
        PorI_arr  = (np.array(ratios)/np.sum(ratios) * np.expand_dims(kwargs[PorI],axis=-1)).T
        
        for i in range(np.array(sidebands).shape[-1]):
            kwargs[PorI] = PorI_arr[i]
            self.add(lamb=lamb, freq_shift=offset_freq+mod_freqs[i], **kwargs)
            #save input parameters offset_freq and mod_freq to be able to look it up later
            self.entries[-1].offset_freq = offset_freq
            self.entries[-1].mod_freq  = mod_freq
    
    def make_retrorefl_beams(self, beam_config='oneside',
                             P_tot=100e-3, FWHM=1.6e-3, w_cylind=0.0,
                             T_airglass=99.62e-2, R_mirror=98.34e-2,
                             mirror_sep=463e-3, reflections=34,
                             int_length=200e-3, x_offset=15e-3, cut_flanks=True,
                             printing=True, plotting=False,
                             **laser_kwargs,
                             ):
        """Create all ``Laser`` objects for a realistic retroreflecting beam
        configuration for a long cooling interaction region in the experiment.

        Parameters
        ----------
        beam_config : str, optional
            'oneside' when only one laser beam is retro-reflected from one side.
            'twosides' for two laser beams entering the interaction region from both
            sides, e.g. for Sisyphus cooling. The default is 'oneside'.
        P_tot : float, optional
            Total inital power (W) of the incoming laser beam or beams. The default is 100e-3.
        FWHM : float, optional
            FWHM of the beams. The default is 1.6e-3.
        w_cylind : float, optional
            width (m) of a cylindrical widened beam. The default is 0.0.
        T_airglass : float, optional
            Single transmission air - glass. The default is 99.62e-2.
        R_mirror : float, optional
            Reflectivity of the mirror. The default is 98.34e-2.
        mirror_sep : float, optional
            Separation (m) between both retro-reflecting mirrors. The default is 463e-3.
        reflections : int, optional
            number of reflections (counting all reflections of a single incoming beam
            on all mirrors). The default is 34.
        int_length : float, optional
            Total length (m) of the interaction region, i.e. from first to last intensity
            peak along the centered axis. The default is 200e-3.
        x_offset : float, optional
            Additional offset of the reflections, i.e. position of the first reflection
            on the centered axis. The default is 15e-3.
        cut_flanks : bool, optional
            Whether the flanks of the beam profiles should be cutted which is usually
            the case on the mirror for two beams from both sides. The default is True.
        printing : bool, optional
            Whether printing additional information. The default is True.
        plotting : bool, optional
            Whether plotting the 1D and 2D intensity distributions. The default is False.
        **laser_kwargs : kwargs, optional
            Further keyword arguments for the created laser objects such as e.g.
            wavelength.
        """
        # longitudinal difference between two reflecting events
        dx          = int_length/(reflections-1)
        beam_angle  = np.arctan(dx/mirror_sep) /(2*np.pi) *360 # angle of beam in degrees
        
        self.retrorefl_beams_kwargs = locals()
        
        if cut_flanks:
            if w_cylind == 0:
                w_cylind = FWHM*0.8493218002880192
            r_cylind_trunc = dx/2#/1.01
        else:
            r_cylind_trunc = 10.
            
        P           = P_tot
        for i in np.arange(reflections):
            # Calculate current laser power due to losses from transmission and reflection
            pm1     = i%2*2 - 1
            if i == 0:
                # first beam pass only one time transmission through glass
                P   *= T_airglass**2 
            else:
                P   *= T_airglass**2 * R_mirror * T_airglass**2  
            # print('P= {:.2f}mW'.format(P*1e3))
            
            # Only one laser without sidebands required since only the total
            # intensities of all lasers would be simply added:
            if beam_config == 'oneside':
                P_beam = P
            elif beam_config == 'twosides':
                P_beam = P/2
            self.add(
                P           = P_beam,
                k           = [dx/mirror_sep, 0, +pm1],
                r_k         = [i*dx+x_offset, 0, 0],
                FWHM        = FWHM,
                w_cylind    = w_cylind,
                r_cylind_trunc= r_cylind_trunc,
                dir_cylind  = [1, 0, -pm1*dx/mirror_sep],
                **laser_kwargs,
                )
            if beam_config == 'twosides':
                self.add(
                    P           = P_beam,
                    k           = [dx/mirror_sep, 0, -pm1],
                    r_k         = [i*dx+x_offset, 0, 0],
                    FWHM        = FWHM,
                    w_cylind    = w_cylind,
                    r_cylind_trunc= r_cylind_trunc,
                    dir_cylind  = [1, 0, +pm1*dx/mirror_sep],
                    **laser_kwargs,
                    )
        if printing:
            print('k=',[dx/mirror_sep,0,+pm1],', dir_cyl=',[1,0,-pm1*dx/mirror_sep])
            print('Power ratio between last and first beam ={:6.2f} %'.format(
                P / (P_tot*T_airglass**2) *1e2 ))
            print(f"Beam angle ={beam_angle:6.3f}° and {self.pNum:3} laser beam objects in total")
            d_travel = np.sqrt(dx**2 + mirror_sep**2)*reflections
            print(f"Travelling distance of a single laser beam ={d_travel:6.2f}m")
            
        if plotting:
            plt.figure('1D intensity distribution')
            z_shift = 0e-3
            self.plot_I_1D(ax='x',axshifts=[0e-3,z_shift],limits=[0e-2,23e-2],
                                  label=f"{reflections} refl.", Npoints=10001)
            # 2D
            plt.figure('2D intensity distribution')
            self.plot_I_2D(ax='y',axshift=0,Npoints=501,
                                  limits=([0e-2,23e-2],[-mirror_sep/2,+mirror_sep/2]))
    
    def get_intensity_func(self,sum_lasers=True,use_jit=True):
        '''Generate a fast function which uses all the current parameters of
        all lasers in this Lasersystem for calculating the total intensity.
        This function can also be called directly by calling the method
        :func:`I_tot` with an input parameter ``r`` as the
        position at which the total intensity is calculated.
        
        Parameters
        ----------
        sum_lasers : bool, optional
            If True, the returned intensity function evaulates the intensities
            of all laser instances for returning the local total intensity sum.
            If False, the returned intensity function only returns an array with
            the length of defined laser instances. This array contains the factors
            which corresponds to the local intensity of each laser divided by
            its maximum intensity at the center of the Gaussian distribution.
            The default is True.
        use_jit : bool, optional
            The returned function can be compiled in time to a very fast C code
            using the numba package. However, the compilation time can be a few
            seconds long the first time the function is called. For all later
            calls it is then much faster. The default is True.

        Returns
        -------
        function
            it's the same function which is used in the method :func:`I_tot`
        '''
        if sum_lasers:
            if self.intensity_func_sum != None:
                return self.intensity_func_sum
        else: 
            if self.intensity_func != None:
                return self.intensity_func
        
        pNum    = self.pNum
        I_arr   = self.getarr('I')
        w       = self.getarr('w')
        w_cyl   = self.getarr('_w_cylind')
        r_cyl_trunc = self.getarr('_r_cylind_trunc')
        dir_cyl = self.getarr('_dir_cylind') #unit vectors
        k       = self.getarr('k') #unit vectors
        r_k     = self.getarr('r_k')
        
        # very fast function which calculates the total intensity only for the
        # parameters which are defined before
        # @jit(nopython=False,parallel=False,fastmath=True,forceobj=True)
        def I_tot(r):
            factors = np.zeros(pNum)
            for p in range(pNum):
                r_ = r - r_k[p]
                if w_cyl[p] != 0.0: # calculation for a beam which is widened by a cylindrical lens
                    d2_w = np.dot(dir_cyl[p],r_)**2
                    if d2_w > r_cyl_trunc[p]**2: #test if position is larger than the truncation radius along the dir_cyl direction
                        continue
                    else:
                        d2 = np.dot(np.cross(dir_cyl[p],k[p]),r_)**2
                        factors[p] = np.exp(-2*(d2_w/w_cyl[p]**2 + d2/w[p]**2))
                else: 
                    r_perp = np.cross( r_ , k[p] )
                    factors[p] = np.exp(-2 * np.dot(r_perp,r_perp) / w[p]**2 )
            if sum_lasers:
                return np.sum(factors*I_arr)
            else:
                return factors
            
        if use_jit:
            I_tot = jit(nopython=True,parallel=False,fastmath=True)(I_tot)
            if sum_lasers:
                self.intensity_func_sum = I_tot
            else:
                self.intensity_func = I_tot
        return I_tot
    
    def I_tot(self,r,**kwargs):
        '''Calculate the total intensity of all lasers in this Lasersystem at
        a specific position `r`. For this calculation the function generated by
        :func:`get_intensity_func` is used.

        Parameters
        ----------
        r : 1D array of size 3
            position at which the total intensity is calculated.
        **kwargs : keywords
            optional keywords of the method :func:`get_intensity_func` can be provided.

        Returns
        -------
        float
            total intensity at the position r.
        '''
        return self.get_intensity_func(**kwargs)(r)
        
    def plot_I_2D(self,ax='x',axshift=0,limits=([-0.05,0.05],[-0.05,0.05]),Npoints=201):
        """Plot the 2D intensity distribution of all laser beams along two axes
        by using the method :func:`get_intensity_func`.
        
        Parameters
        ----------
        ax : str, optional
            axis orthogonal to the plane to be plotted. Can be 'x','y' or 'z'.
            The default is 'x'.
        axshift : float, optional
            shift along the axis `ax` which defines the absolute position of
            the plane to be plotted. The default is 0.
        limits : tuple(list,list), optional
            determines the minimum and maximum limit for both axes which lies
            in the plane to be plotted.
            The default is ([-0.05,0.05],[-0.05,0.05]).
        Npoints : int, optional
            Number of plotting points for each axis. The default is 201.
        """
        axshift = float(axshift)
        xyz = {'x':0,'y':1,'z':2}
        ax_ = xyz[ax]
        del xyz[ax]
        axes_ = np.array([*xyz.values()])
        lim1,lim2 = limits
        x1,x2 = np.linspace(lim1[0],lim1[1],Npoints),np.linspace(lim2[0],lim2[1],Npoints)
        Z = np.zeros((len(x1),len(x2)))
        r = np.zeros(3)
        for i in range(Npoints):
            for j in range(Npoints):
                r[ax_] = axshift
                r[axes_] = x1[i],x2[j]
                Z[i,j] = self.I_tot(r,sum_lasers=True,use_jit=True)
        
        X1,X2 = np.meshgrid(x1,x2)
        # plt.figure('Intensity distribution of all laser beams at {}={:.2f}mm'.format(
            # ax,axshift*1e3))
        plt.contourf(X1*1e3,X2*1e3,Z.T,levels=20)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Intensity $I_{tot}$ in W/m$^2$')
        keys = list(xyz.keys())
        plt.xlabel('position {} in mm'.format(keys[0]))
        plt.ylabel('position {} in mm'.format(keys[1]))
    
    def plot_I_1D(self,ax='x',axshifts=[0,0],limits=[-0.05,0.05],
                  Npoints=1001,label=None):
        """Plot the 1D intensity distribution of all laser beams along an axis
        by using the method :func:`get_intensity_func`.

        Parameters
        ----------
        ax : str, optional
            axis along which the intensity distribution is plotted. The default is 'x'.
        axshifts : list, optional
            shifts in m of the other two axes besides `ax`. The default is [0,0].
        limits : list, optional
            determines the minimum and maximum limit for the axis `ax`.
            The default is [-0.05,0.05].
        Npoints : int, optional
            Number of plotting points along the axis `ax`. The default is 1001.
        label : str, optional
            label for the plotted curve. If None, the label shows the values of
            `axshifts`. The default is None.
        """
        axshifts = np.array(axshifts,dtype=float) # shifting other two axes with offset
        xyz = {'x':0,'y':1,'z':2}
        ax_ = xyz[ax] # index of the axis on which we want to plot the intensity
        del xyz[ax]
        axes = list(xyz.keys())
        axes_ = np.array([*xyz.values()])
        
        # plt.figure('Intensity over x')
        x_arr = np.linspace(limits[0],limits[1],Npoints)
        y_arr = np.zeros(Npoints)
        
        r = np.zeros((Npoints,3))
        r[:,axes_] += axshifts
        r[:,ax_]    = x_arr
        for i,r_i in enumerate(r):
            y_arr[i] = self.I_tot(r_i,sum_lasers=True,use_jit=True)
            
        if label == None:
            label = '{}={:.2f}mm, {}={:.2f}mm'.format(axes[0],axshifts[0]*1e3,axes[1],axshifts[1]*1e3)
        plt.plot(x_arr*1e3,y_arr,label=label)
        plt.legend()
        plt.xlabel('position {} in mm'.format(ax))
        plt.ylabel('Intensitiy $I$ in W/m$^2$')
    
    def _get_wavelength_regimes(self, subplot_sep):
        # sorted frequencies
        fsort   = np.sort(np.atleast_2d( self.getarr('f').T )[0])
        
        # indices of laser blocks that are separated by the frequency subplot_sep
        inds    = [i+1 for i,df in enumerate(np.diff(fsort)) if df > subplot_sep]
        inds    = [0, *inds]
    
        # mean of each block of frequencies
        fmean   = np.array([ fsort[i:j].mean() for i,j in zip(inds, inds[1:]+[None])])
        
        return list(c/fmean) # wavelengths
    
    def plot_spectrum(self, axs=[], unit='MHz', subplot_sep=1e9, std=0, wavelengths=[],
                   invert=False, N_points=500, xaxis_ext=5, cmap=None, relative_to_wavelengths=False,
                   fill_between_kwargs=dict(alpha=0.2, color='grey'),
                   plot_kwargs=dict(color='grey',ls='-')):
        """Plot the spectrum of :class:`~.Lasersystem.Laser` objects
        with their respective intensities.
        Either as Gaussians for each single laser object or as vertical lines.
        
        Note
        ----
        This method also supports plotting multiple detunings with different colors
        from ``matplotlib.colormap`` into a single subplot.

        Parameters
        ----------
        axs : list of ``matplotlib.pyplot.axis`` objects, optional
            axis/axes to put the plot(s) on. The default is [].
        unit : str, optional
            Unit of the x-axis to be plotted.
            Can be one of ``['GHz','MHz','kHz','Hz']``. Default is 'MHz'.
        subplot_sep : float, optional
            Defines the range of the plotted x-axis and the separation for the 
            automatic inclusion of all wavlengths (see parameter ``wavelengths``)
            in Hz. The default is 1e9.
        std : float, optional
            Standard deviation of the individual Gaussians to be drawn in MHz.
            The default is 0 meaning that instead of Gaussians, vertical lines
            are drawn.
        wavelengths : list, optional
            wavelengths that should be plotted within the range ``subplot_sep``.
            By default all available laser wavelengths are used.
        invert : bool, optional
            Whether the plot y-axis should be inverted. The default is False.
        N_points : int, optional
            number of points plotted. The default is 500.
        xaxis_ext : TYPE, optional
            The range of the xaxis, given by the lowest and highest transition
            frequency, is extended by this factor multiplied by the sum of the
            Gaussian broadening (``std``). The default is 5.
        cmap : std, optional
            ``matplotlib.colormap`` for plotting multiple spectra for multiple
            detunings. The default is None.
        relative_to_wavelengths : bool, optional
            Whether the x-axis should be plotted in absolute frequency units or
            relative to ``wavelengths``. The default is False.
        fill_between_kwargs : dict, optional
            kwargs for the method ``matplotlib.pyplot.fill_between``.
            For disabling this plotting, just provide an empty dictionary or False.
            The default is dict(alpha=0.2, color='grey').
        plot_kwargs : dict, optional
            kwargs for the method normal line plotting method ``matplotlib.pyplot.plot``.
            For disabling this plotting, just provide an empty dictionary or False.
            The default is dict(color='grey',ls='-').

        Returns
        -------
        axs : list of ``matplotlib.pyplot.axis``
            Axes of the subplot(s).
        """
        
        # get wavelengths of all laser wavelength blocks that span subplot_sep*Gamma
        if not wavelengths:
            wavelengths = self._get_wavelength_regimes(subplot_sep=subplot_sep)
        
        # check iterating laser parameters
        iters = self._identify_iter_params()[0]
        if 'omega' in iters:
            iters.pop('omega')
        if iters:
            raise Exception(
                (f"The iterating parameters {list(iters.keys())} of laser "
                "objects is not supported for plotting (except for 'omega')."))

        # generate subplots
        axs = tools.auto_subplots(len(wavelengths), axs=axs, 
                                  ylabel=f'Intensity (W/m$^2$)')
        f_arr   = np.atleast_2d( self.getarr('f').T )
        # scaling x
        fac_x = {'GHz':1e9, 'MHz':1e6, 'kHz':1e3, 'Hz':1}[unit]
        
        for ax,wavelength in zip(axs,wavelengths):
            
            xconv   = lambda x: (x - c/wavelength*int(bool(relative_to_wavelengths)))/fac_x
            yconv   = lambda y: y
            
            inds    = np.argwhere((f_arr[0]>c/wavelength-subplot_sep) \
                                  & (f_arr[0]<c/wavelength+subplot_sep))[:,0]
            
            for j,f_arr_i in enumerate(f_arr):
                
                if f_arr.shape[0] != 1:
                    color   = plt.get_cmap(cmap)(j/(len(f_arr)-1))
                    fill_between_kwargs.update(color=color)
                    
                f_cut   = f_arr_i[inds]
                spectrum = []
                for i in inds:
                    la = self[i]
                    if std:
                        x_ext   = std*xaxis_ext # extension on the x axis
                        xaxis = np.linspace(min(f_cut)-x_ext, max(f_cut)+x_ext, N_points)
                        y = tools.gaussian(xaxis, a=la.I, x0=f_arr_i[i], std=std, y_off=0)
                        spectrum.append(y)
                    else:
                        ax.plot(2*[xconv(f_arr_i[i])], [0, yconv(la.I)],
                                lw=2, **plot_kwargs)
                        # ax.axvline((la.f-c/wavelength)/fac_x, ymin=0, ymax=la.I/Inorm,
                        #            color='grey', lw=2, alpha=1)
                if std:
                    spectrum = np.sum(spectrum, axis=0)
                    if fill_between_kwargs:
                        ax.fill_between(xconv(xaxis), 0, yconv(spectrum), **fill_between_kwargs)
                    if plot_kwargs:
                        ax.plot(xconv(xaxis), yconv(spectrum),
                                **plot_kwargs)
            xlabel = f'Frequency ({unit})'
            if relative_to_wavelengths:
                xlabel += f" at {wavelength*1e9:f} nm"
            ax.set_xlabel(xlabel)
            if invert:
                ax.yaxis.set_inverted(True)  # inverted axis with autoscaling
            
        return axs
    
    def __delitem__(self,index):
        """delete lasers using del system.lasers[<normal indexing>], or delete all del system.lasers[:]"""
        #delete lasers with del system.lasers[<normal indexing>], or delete all del system.lasers[:]
        del self.entries[index]
        self.intensity_func = None
        self.intensity_func_sum = None
        
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice,np.integer)): 
            return self.entries[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.entries[i] for i in index]
    
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        with pd.option_context(*self.pd_display_options):
            return f"{self.description}\n{self.DF()}"
    
    def __repr__(self):
        with pd.option_context(*self.pd_display_options):
            return repr(self.DF())
    
    def _repr_html_(self):
        with pd.option_context(*self.pd_display_options):
            return self.DF()._repr_html_()
        
    def DF(self, **kwargs):       
        """
        Create a pretty :class:`pandas.DataFrame` object with all lasers and
        their attributes. This method basically concatenates the individual
        laser dataframes from :class:`Laser.DF()`.
        This method is e.g. used when calling ``print(lasers)``.

        Parameters
        ----------
        **kwargs : kwargs
            Keyword arguments of the respective method :meth:`Laser.DF()`.

        Returns
        -------
        pandas.DataFrame
            Dataframe with the lasers attributes.
        """
        if self.pNum == 0:
            return None
        else:
            return pd.concat([la.DF().T for la in self],
                             ignore_index=True)
        
    def check_config(self,raise_Error=False):
        """Check the configuration for simulating internal dynamics using
        the OBEs or rate equations.
        """
        if self.pNum == 0:
            Err_str = 'There are no lasers defined!'
            if raise_Error: raise Exception(Err_str)
            else: warnings.warn(Err_str)
        #maybe also check if some dipole matrices are completely zero or
        # if the wavelengths are in wrong order of magnitude??
        
    @property
    def description(self):
        """
        Display a short description with the number of included laser objects.
        
        Returns
        -------
        str
            description of the lasersystem.
        """
        return "{:d} - Lasersystem".format(self.pNum)
    
    @property
    def pNum(self):
        """int: Calculate the number of included Laser objects."""
        return len(self.entries)
    
    @property
    def I_sum(self):
        """
        Calculate the sum of the peak intensities of all laser beams

        Returns
        -------
        numpy.ndarray or float
            Sum of peak intensities.
        """
        return np.array([la.I for la in self]).sum(axis=0)
    
    @property
    def P_sum(self):
        """
        Calculate the sum of the powers of all laser beams

        Returns
        -------
        numpy.ndarray or float
            Sum of all lasers' powers.
        """
        return np.array([la.P for la in self]).sum()
    
#%%
class Laser:
    #: units of the laser beam properties
    UNITS = dict(
        lamb    = 'm',
        I       = 'W/m^2',
        P       = 'W',
        FWHM    = 'm',
        k       = '1',
        r_k     = 'm',
        phi     = 'rad',
        beta    = 'Hz/s',
        f_q     = '1',
        w       = 'm',
        _w_clind= 'm',
        _dir_clind='m',
        )
    
    def __init__(self,lamb=860e-9,freq_shift=0,pol='lin',pol_direction=None,
                 P=20e-3,I=None,FWHM=5e-3,w=None,
                 w_cylind=.0,r_cylind_trunc=5e-2,dir_cylind=[1,0,0],
                 freq_Rabi=None,k=[0,0,1],r_k=[0,0,0],beta=0.,phi=0.0,
                 pol2=None,pol2_direction=None):
        """This class contains all physical properties of a laser which can
        be assembled in the class :class:`Lasersystem`.
        
        Note
        ----
        ``freq_shift`` is given as non-angualar frequency, i.e. without the
        :math:`2 \pi` factor.
        
        Parameters
        ----------
        lamb : float, optional
            wavelength lambda. The default is 860e-9.
        freq_shift : float, optional
            Shift of the laser's frequency (without 2 pi) additional to the
            frequency determined by Parameter lamb. The default is 0.0.
        pol : str, tuple(str,str), optional
            polarization of the laserbeam. Can be either 'lin', 'sigmap' or
            'sigmam' for linear or circular polarized light of the laser.
            For polarization switching a tuple of two polarizations is needed.
            The default is 'lin'.
        pol_direction : str, optional
            optional addition to the ``pol`` parameter to be considered in the
            OBEs calculation. Can be either 'x','y','z' for linear polarization
            or 'xy','xz','yz' for circular polarization. Given the default value
            None the linear polarization is aong the quantization axis 'z'
            and the circular ones in 'xy'.
        P : float, optional
            Laser power in W. The default is 20e-3.
        I : float, optional
            Intensity of the laser beam. When specified a given power P is
            ignored. The default is None.
        FWHM : float, optional
            FWHM (full width at half maximum) of the Gaussian intensity
            distribution of the laserbeam. When this value is adjusted after
            the initialization of the object the w value is automatically
            corrected but to further adjust the intensity the power has to be
            set again. The default is 5e-3.
        w : float, optional
            :math:`1/e^2` beam radius of the Gaussian intensity distribution.
            When this value is adjusted after the initialization of the object
            the FWHM value is automatically corrected but to further adjust the
            intensity the power has to be set again. The default is None.
        w_cylind : float, optional
            :math:`1/e^2` beam radius of the Gaussian intensity distribution
            along x direction for the specific configuration where the
            laser beam is aligned in y axis direction and has a widened intensity
            distribution along x axis with radius `w_cylind`. The distribution
            along the z axis is given by the radius `w`.
            The default is 0.0.
        r_cylind_trunc : float, optional
            specifies the radial distance along the direction `dir_cylind`
            (widened by a cylindrical lens) at which the intensity is truncated.
            The default is 5e-2.
        dir_cylind : 1D array of size 3, optional
            Direction in which the beam is widened by a cylindrical lens.
            This direction has to be orthogonal to the laser wave vector `k`.
            This variable has only an effect when the input parameter
            `w_cylind` is non-zero. The default is [1,0,0].
        freq_Rabi : float, optional
            Rabi frequency in terms of angular frequency 2 pi. The appropriate
            intensity is first set to an arbitrary value since it is adjusted
            later during the calculation where the levels are involved.
            The default is None.            
        k : list or array type of dimension 3, optional
            direction of the wave vector :math:`\hat{k}` of the laserbeam.
            The inserted array is automatically normalized to unit vector.
            The default is [0,0,1].
        r_k : list or array type of dimension 3, optional
            a certain point which is located anywhere within the laserbeam.
            The default is [0,0,0].
        beta : float, optional
            When the frequency of the laser should be varied linearly in time,
            then `beta` defines the chirping rate in Hz/s (without factor of 2 pi).
            The default is 0.0.
        phi : float, optional
            phase offset of the laser's electric field in rad (important e.g.
            for standing waves). The default is 0.0.

        Raises
        ------
        Exception
            When the given type of the ``pol`` Parameter is not accepted.
            
        Example
        -------
        A fast way to calculate the power of a laser with certain beam radii
        to reach a certain intensity (or the other way around for an intensity)::
            
            from MoleCool import Laser
            print( Laser(I = 1000., w = 1e-3, w_cylind = 5e-2).P )
            print( Laser(P = 0.02,  FWHM = 5e-3).I )
        """
        #: float: angular frequency :math:`\omega`
        self.omega      = 2*pi*(c/lamb + freq_shift)
        # different quantities when a cylindrical lens is used widening the laser beam along one transversal axis
        self._w_cylind, self._r_cylind_trunc = w_cylind, r_cylind_trunc
        self._dir_cylind = np.array(dir_cylind)/np.expand_dims(np.linalg.norm(dir_cylind,axis=-1),axis=-1) #unit vector
        #___definition of the beam width:
        #   if a 1/e^2 radius is given. It is used for further calculations. Otherwise the FWHM value is used.
        if np.any(w != None):
            self.w = w # old **default** value: (2*(pi*1.5e-3**2))**0.5 --> arbitrary value to compare to old MATLAB rate equations
        elif np.any(FWHM != None):
            self.FWHM = FWHM
        #___intensity definition or calculation via P and beam widths w & w_cylind:
        #: Rabi frequency in terms of angular frequency 2 pi
        self.freq_Rabi = freq_Rabi
        if np.any(freq_Rabi != None):
            self.I  = 1.0 #arbitrarily setting initial value for intensity since it is adjusted later during the calculation where the levels are involved.
            self._P = None
        # intensity I is important quantity for calculations instead of the power P.
        elif np.any(I != None):
            self.I  = I
            self._P = None
        else:
            self.P  = P #calculation of the intensity using the power and beam widths.
        
        #: unit wavevector :math:`\hat{k}`
        self.k      = np.array(k)/np.expand_dims(np.linalg.norm(k,axis=-1),axis=-1) #unit vector
        if (w_cylind != 0.0) and (np.dot(self._dir_cylind,self.k) != 0.0):
            raise Exception('input variable dir_cylind has to be orthogonal to the wave vector k')
        #: any point which is passed by the laser wave vector (i.e. the point lying in the propagation line of the laser)
        self.r_k    = np.array(r_k) #point which is lying in the laserbeam
        #: laser chirping rate for linear varying the laser frequency in time
        self.beta   = beta
        #: phase offset of the laser's electric field (important e.g. for standing waves)
        self.phi    = phi
        
        #___define the laser polarizations (and polarization direction)
        self.f_q = self._get_polarization_comps(pol,pol_direction)
        if pol2 != None:
            self.pol_switching  = True
            self.f_q2           = self._get_polarization_comps(pol2,pol2_direction)
        else:
            self.pol_switching  = False
            self.f_q2           = self.f_q.copy()   
        
    def _get_polarization_comps(self,pol,pol_direction):
        # check if pol has the right datatype and then if it has the right value
        if type(pol) != str:
            raise Exception("Wrong datatype or length of <pol>: only str allowed")
        pol_list = ['lin','sigmap','sigmam']
        if not (pol in pol_list):
            raise Exception("'{}' is not valid for <pol>, it can only be '{}','{}', or '{}'".format(pol,*pol_list))
        
        if pol_direction == None:
            if pol == 'lin':      f_q = np.array([0.,1.,0.]) #q= 0; mF -> mF'= mF
            elif pol == 'sigmam': f_q = np.array([0.,0.,1.]) #q=+1; mF -> mF'= mF-1
            elif pol == 'sigmap': f_q = np.array([1.,0.,0.]) #q=-1; mF -> mF'= mF+1
        else:
            p = pol_direction
            x = np.array([+1., 0,-1.])/np.sqrt(2)
            y = np.array([+1., 0,+1.])*1j/np.sqrt(2)
            z = np.array([ 0, +1, 0 ])
            if isinstance(p,(list,np.ndarray)):
                f_q = p[0]*x + p[1]*y + p[2]*z # not yet programmed in the best way!?
            elif isinstance(p,str):
                if len(p) == 1:
                    if p == 'x':   f_q = x
                    elif p == 'y': f_q = y
                    elif p == 'z': f_q = z
                if len(p) == 2:
                    if pol == 'sigmam':
                        a1,a2 = -1., -1j
                    elif pol == 'sigmap':
                        a1,a2 = +1., -1j
                    if p == 'xy':   f_q = a1*x + a2*y
                    elif p == 'xz': f_q = a1*z + a2*x
                    elif p == 'yz': f_q = a1*y + a2*z
            else: #maybe also check if the string values of pol_direction is correct?!
                raise Exception("Wrong datatype of <pol_direction>")
        return np.array([ -f_q[2], +f_q[1], -f_q[0] ]) / np.linalg.norm(f_q)
            
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        with pd.option_context('display.float_format','{:.3e}'.format):
            return self.DF().to_string()
    
    def __repr__(self):
        with pd.option_context('display.float_format','{:.3e}'.format):
            return repr(self.DF())
    
    def _repr_html_(self):
        with pd.option_context('display.float_format','{:.3e}'.format):
            return self.DF()._repr_html_()
    
    def DF(self,
           attrs  = ['lamb','I','P','FWHM','k','r_k','phi','beta'],
           units = {},
           ):
        """
        Create a DataFrame with attributes of the Laser instance. This method
        is e.g. used when calling ``print(lasers[0])``.
        
        Parameters
        ----------
        attrs : list, optional
            list of attributes as properties of the Laser.
            The default is ['lamb', 'I', 'P', 'FWHM', 'k',
            'r_k', 'phi', 'beta'].
        units : dict, optional
            units of the physical attributes. The default is {}.

        Returns
        -------
        df : pandas.DataFrame
            All specified attributes with units as a Dataframe.
        """
        
        units = dict(self.UNITS, **units)
        for attr in attrs:
            if not hasattr(self, attr):
                raise ValueError(f"attr {attr} is not an attribute of Laser")
            if attr not in units:
                units[attr] = '?'
        
        df = pd.DataFrame(
            [getattr(self, attr) for attr in attrs],
            index = [f"{attr} ({units[attr]})" for attr in attrs]
            )
        
        return df
    
    @property
    def w(self):
        """Calculate the 1/e^2 beam radius"""
        return self._w
    
    @w.setter
    def w(self,w):
        self._w = w
        self._FWHM = 2*w / ( np.sqrt(2)/np.sqrt(np.log(2)) )
        self.intensity_func = None
        self.intensity_func_sum = None
        
    @property
    def FWHM(self):
        """Calculate the  FWHM (full width at half maximum) of the Gaussian
        intensity distribution of the laserbeam
        """
        return self._FWHM
    
    @FWHM.setter
    def FWHM(self,FWHM):
        self._FWHM = FWHM
        self._w = np.sqrt(2)/np.sqrt(np.log(2))*FWHM/2 # ~= 1.699*FWHM/2
        self.intensity_func = None
        self.intensity_func_sum = None
        
    @property
    def P(self):
        """Calculate the Power of the single beam"""
        if np.any(self._P != None): return self._P
        else:
            if np.any(np.array(self._w_cylind) != 0.0):
                return self.I*(pi*self.w*self._w_cylind)/2
            else: return self.I*(pi*self.w**2)/2
            
    @P.setter
    def P(self,P):
        """When the power P is set to a value the intensity is automatically
        calculated using the beam widths."""
        self._P = P
        if np.any(np.array(self._w_cylind) != 0.0):
            self.I  = 2*self.P/(pi*self.w*self._w_cylind)
        else:
            #: float: :math:`I =P/A` with the Area :math:`A=\pi w_1 w_2/2` of a 2dim Gaussian beam
            self.I  = 2*self.P/(pi*self.w**2)
        self.intensity_func = None
        self.intensity_func_sum = None
        
    @property
    def kabs(self):
        """Calculate the absolute value of the wave vector
        (:math:`= 2 \pi/\lambda = \omega/c`)
        in :math:`\\text{rad}/\\text{m}`.
        
        Note:
            ``self.k`` is a unit vector and defines the direction of the wave vector"""
        return self.omega/c
    
    @property
    def lamb(self):
        """Calculate the wavelength of the single laser"""
        return 2*pi*c/self.omega
    
    @property
    def f(self):
        """Calculate the frequency (non-angular)"""
        return self.omega/(2*pi)
    
    @property
    def E(self):
        """Energy of the laser's photons."""
        return self.omega * hbar