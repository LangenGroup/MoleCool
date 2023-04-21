# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:34:09 2020

@author: fkogel

v3.2.0

This module contains all classes and functions to define a System including
multiple :class:`Laser` objects.

Example
-------
Below an empty Lasersystem is created and a single Laser with wavelength 860nm
and Power 20 mW with linear polarization is added::
    
    lasers = Lasersystem()
    lasers.add(860e-9,20e-3,'lin')

But first start python and import the module::
    
    $ python
    >>> import Lasersystem
    
Tip
---
Every object of the classes :class:`Lasersystem` or :class:`Laser` class can
be printed to display all attributes via::
    
    print(lasers)
    print(lasers[0])
    
To delete all instances use this command::
    
    del lasers[:]
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g
from numba import jit
import matplotlib.pyplot as plt
import warnings
#%%
class Lasersystem:
    def __init__(self,freq_pol_switch=5e6):
        """System consisting of :py:class:`Lasersystem.Laser` objects
        and methods to add them properly.
        These respective objects can be retrieved and also deleted by using the
        normal item indexing of a :class:`Lasersystem`'s object::
            
            lasers = Lasersystem()
            lasers.add(lamb=860e-9,P=20e-3,pol='lin')
            lasers.add(lamb=890e-9,I=1000,FWHM=2e-3)            
            laser1 = lasers[0] # call first Laser object included in lasers
            del lasers[-1] # delete last added Laser object
        
        Within the command in the first line an empty `self.entries` list is
        created to store all :class:`Laser` objects.
        
        Example
        -------
        ::
            
            lasers = Lasersystem()
            lasers.add_sidebands(lamb=860e-9,P=20e-3,pol='lin',offset_freq=20e6,mod_freq=39e6)
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

    def add(self,lamb=860e-9,P=20e-3,pol='lin',**kwargs):
        """adds an instance of :class:`Laser` to this class. 
        
        Note
        ----
        Is the same as:
            >>> self.entries.append(Laser(...)).
        
        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments. Same as in the ``__init__`` method of
            the class :class:`Laser` (further information)
        """
        self.entries.append( Laser( lamb=lamb, P=P, pol=pol, **kwargs) ) 
        self.intensity_func = None
        self.intensity_func_sum = None
    
    def getarr(self,var):
        self.check_config()
        if not var in dir(self[0]):
            raise ValueError('The attribute {} is not included in the Laser objects'.format(var))
        if var == 'f_q': 
            dtype = complex
        else:
            dtype = float
        return np.array([getattr(la,var) for la in self],dtype=dtype)
        
    def add_sidebands(self,lamb=860e-9,offset_freq=0.0,mod_freq=1e6,
                      ratios=None,sidebands=[-1,1],**kwargs):
        """Adds ``Laser`` instances as sidebands in order to drive multiple
        hyperfine transitions. The individual sidebands are detuned from the center
        frequency by the modulation frequency `mod_freq` times the values in the
        list `sidebands`, i.e. for `mod_freq=1e6` and `sidebands=[-1,0,2]`, the
        sidebands are detuned by -1 MHz, 0 MHz and 2 MHz. The center frequency
        is given by the wavelength lamb and an additional general offset frequency
        `offset_freq`.
        
        Parameters
        ----------
        lamb : :py:obj:`float`
            wavelength of the main transition.
        P : `float`
            Power, i.e. sum of the powers of all sidebands.
            Alternativley the sum of the intensities can be provided.
        I : `float`
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
            optional arguments  (see :class:`Laser`).
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
        PorI_arr  = np.array(ratios)/np.sum(ratios) * np.expand_dims(kwargs[PorI],axis=-1)
        
        for i in range(len(sidebands)):
            kwargs[PorI] = (PorI_arr.T)[i]
            self.add(lamb=lamb, freq_shift=offset_freq+mod_freqs[i], **kwargs)
            #save input parameters offset_freq and mod_freq to be able to look it up later
            self.entries[-1].offset_freq = offset_freq
            self.entries[-1].mod_freq  = mod_freq
    
    def get_intensity_func(self,sum_lasers=True,use_jit=True):
        '''generates a function which uses all the current parameters of all
        lasers in this Lasersystem for calculating the total intensity.
        This function can also be called directly by calling the method
        :func:`I_tot` with an input parameter r as
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
        '''calculates the total intensity of all lasers in this Lasersystem at
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
        """plot the 2D intensity distribution of all laser beams along two axes
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
        """plot the 1D intensity distribution of all laser beams along an axis
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
        """__str__ method is called when an object of a class is printed with print(obj)"""
        for i in range(self.pNum):
            la = self.entries[i]
            print('>>> Laserbeam {:2d}: {}'.format(i,la))
        return self.description
    
    def check_config(self,raise_Error=False):
        if self.pNum == 0:
            Err_str = 'There are no lasers defined!'
            if raise_Error: raise Exception(Err_str)
            else: warnings.warn(Err_str)
        #maybe also check if some dipole matrices are completely zero or
        # if the wavelengths are in wrong order of magnitude??
        
    @property
    def description(self):
        """str: Displays a short description with the number of included laser objects."""
        return "{:d} - Lasersystem".format(self.pNum)
    @property
    def pNum(self):
        """int: returns the number of included Laser objects."""
        return len(self.entries)
    @property
    def I_sum(self):
        """returns the sum of the peak intensities of all laser beams"""
        return np.array([la.I for la in self]).sum(axis=0)
    @property
    def P_sum(self):
        """returns the sum of the powers of all laser beams"""
        return np.array([la.P for la in self]).sum()
    
#%%
class Laser:
    name = None #cooling / repumping laser
    def __init__(self,lamb=860e-9,freq_shift=0,pol='lin',pol_direction=None,
                 P=20e-3,I=None,FWHM=5e-3,w=None,
                 w_cylind=.0,r_cylind_trunc=5e-2,dir_cylind=[1,0,0],
                 freq_Rabi=None,k=[0,0,1],r_k=[0,0,0],beta=0.,phi=0.0,
                 pol2=None,pol2_direction=None):
        """Containing all properties of a laser which can be assembled in the
        Lasersystem class.
        
        Note
        ----
        freq_shift without 2pi factor
        
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
        to reach a certain intensity (or the other way around for an intensity):
            
            >>> print(Laser(I=1000.,w=1e-3,w_cylind=5e-2).P)
            >>> print(Laser(P=0.02,FWHM=5e-3).I)
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
        list1=dir(self).copy()
        out = ''
        for el in list1.copy():
            if el[0]=='_': list1.remove(el)
        for el in list1:
            out+='{}='.format(el)
            value = self.__getattribute__(el)
            if isinstance(value,(float,np.float64)): out+= '{:.2e}, '.format(value)
            elif isinstance(value,(list,np.ndarray)) and len(value) >5: out+= '{}..., '.format(value[:5])
            else: out+= '{}, '.format(value)
        #'lamb={:.2e}, I={:.2e}, P={:.2e}, FWHM={:.2e} ,f={:.2e}, pol={}, pol_switching={}'.format(self.lamb,self.I,self.P,self.FWHM,(self.pol,self.pol2),self.pol_switching)    
        return out[:-2]
    
    @property
    def w(self):
        """calculates the 1/e^2 beam radius"""
        return self._w
    @w.setter
    def w(self,w):
        self._w = w
        self._FWHM = 2*w / ( np.sqrt(2)/np.sqrt(np.log(2)) )
        self.intensity_func = None
        self.intensity_func_sum = None
    @property
    def FWHM(self):
        """calculates the  FWHM (full width at half maximum) of the Gaussian
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
        """calculates the Power of the single beam"""
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
        """calculates the absolute value of the wave vector
        (:math:`= 2 \pi/\lambda = \omega/c`)
        in :math:`\\text{rad}/\\text{m}`.
        
        Note:
            ``self.k`` is a unit vector and defines the direction of the wave vector"""
        return self.omega/c
    @property
    def lamb(self):
        """calculates the wavelength of the single laser"""
        return 2*pi*c/self.omega
    @property
    def f(self):
        """calculates the frequency (non-angular)"""
        return self.omega/(2*pi)
    @property
    def E(self):
        """Energy of the laser's photons."""
        return self.omega * hbar