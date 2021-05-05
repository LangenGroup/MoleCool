# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:34:09 2020

@author: fkogel

v2.5.4

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
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g
from numba import jit
import matplotlib.pyplot as plt
#%%
class Lasersystem:
    def __init__(self,freq_pol_switch=5e6):
        """System consisting of :py:class:`Lasersystem.Laser` objects
        and methods to add them properly.
        These respective objects can be retrieved and also deleted by using the
        normal item indexing of a :class:`Lasersystem`'s object::
            
            lasers = Lasersystem()
            lasers.add(lamb=860e-9,P=20e-3,pols='lin')
            lasers.add(lamb=890e-9,I=1000,FWHM=2e-3)            
            laser1 = lasers[0] # call first Laser object included in lasers
            del lasers[-1] # delete last added Laser object
        
        Within the command in the first line an empty `self.entries` list is
        created to store all :class:`Laser` objects.
        
        Example
        -------
        ::
            
            lasers = Lasersystem()
            lasers.add_sidebands(lamb=860e-9,P=20e-3,pols='lin',AOM_shift=20e6,EOM_freq=39e6)
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

    def add(self,lamb=860e-9,P=20e-3,pols='lin',**kwargs):
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
        self.entries.append( Laser( lamb=lamb, P=P, pols=pols, **kwargs) ) 
        self.intensity_func = None
        
    def add_sidebands(self,lamb=860e-9,P=20e-3,pols='lin',AOM_shift=19.0e6,
                      EOM_freq=39.33e6,ratios=[0.8,1,1,0.8],**kwargs):
        """Adds four ``Lasers`` as sidebands (shifted by a certain AOM and EOM
        frequencies) instead of a single ``Laser`` object to the ``Lasersystem``.
        
        Parameters
        ----------
        lamb : :py:obj:`float`
            wavelength of the main transition.
        P : `float`
            Power of the Laser, i.e. sum of the powers of all sidebands.
            Alternativley the sum of the intensities can be provided.
        pols : str, tuple(str,str)
            polarization of the Laser beams.
        AOM_shift : float
            the center Laser frequency is shifted by an additional AOM shift 
            (without 2 pi).
        EOM_freq : float
            starting from the AOM-shifted main frequency, 4 sideband Laserobjects
            are added with shifted frequencies `[-2,-1,1,2]*EOM_freq` (without 2 pi).
        ratios : array_like, optional
            ratios of the sidebands in the same order as the `EOM_freq` parameter.
            (Will be normed to specify the individual sideband powers).
            The default is [0.8,1,1,0.8].
        **kwargs
            optional arguments  (see :class:`Laser`).
        """
        EOM_freqs   = (np.array([-2,-1,1,2])*np.expand_dims(EOM_freq,axis=-1)).T
        if 'I' in kwargs:
            I_red   = np.array(ratios)/np.sum(ratios) * np.expand_dims(kwargs['I'],axis=-1)
            del kwargs['I']
            for i in range(4):
                self.entries.append(Laser( lamb=lamb, I=(I_red.T)[i], pols=pols,
                    freq_shift=AOM_shift+EOM_freqs[i], **kwargs) )
                #save input parameters AOM_shift and EOM_freq to be able to look it up later
                self.entries[-1].AOM_shift = AOM_shift
                self.entries[-1].EOM_freq  = EOM_freq
        else:
            P_red   = np.array(ratios)/np.sum(ratios) * np.expand_dims(P,axis=-1)
            for i in range(4):
                self.entries.append(Laser( lamb=lamb, P=(P_red.T)[i], pols=pols,
                    freq_shift=AOM_shift+EOM_freqs[i], **kwargs) )
                self.entries[-1].AOM_shift = AOM_shift
                self.entries[-1].EOM_freq  = EOM_freq
        self.intensity_func = None
    
    def get_intensity_func(self):
        '''generates a function which uses all the current parameters of all
        lasers in this Lasersystem for calculating the total intensity.
        This function can also be called directly by calling the method
        :func:`I_tot` with an input parameter r as
        position at which the total intensity is calculated.
        The function is jitted with the numba package to be much faster.

        Returns
        -------
        function
            is the same as :func:`I_tot`
        '''
        pNum    = self.pNum
        I_arr   = np.array([la.I for la in self])
        w       = np.array([la.w for la in self])
        w_cyl   = np.array([la._w_cylind for la in self])
        r_cyl_trunc = np.array([la._r_cylind_trunc for la in self])
        dir_cyl = np.array([la._dir_cylind for la in self],dtype=float) #unit vectors
        k       = np.array([la.k for la in self],dtype=float) #unit vectors
        r_k     = np.array([la.r_k for la in self],dtype=float)
        
        # very fast function which calculates the total intensity only for the
        # parameters which are defined before
        @jit(nopython=True,parallel=False,fastmath=True)
        def I_tot(r):
            I=0.0
            for p in range(pNum):
                r_ = r - r_k[p]
                if w_cyl[p] != 0.0: # calculation for a beam which is widened by a cylindrical lens
                    d2_w = np.dot(dir_cyl[p],r_)**2
                    if d2_w > r_cyl_trunc[p]**2: #test if position is larger than the truncation radius along the dir_cyl direction
                        continue   
                    else:
                        d2 = np.dot(np.cross(dir_cyl[p],k[p]),r_)**2
                        I += I_arr[p] *np.exp(-2*(d2_w/w_cyl[p]**2 + d2/w[p]**2))
                else: 
                    r_perp = np.cross( r_ , k[p] )
                    I += I_arr[p] *np.exp(-2 * np.dot(r_perp,r_perp) / w[p]**2 )   
            return I
        self.intensity_func = I_tot
        return I_tot
    
    def I_tot(self,r):
        '''calculates the total intensity of all lasers in this Lasersystem at
        a specific position `r`. For this calculation the function generated by
        :func:`get_intensity_func` is used.

        Parameters
        ----------
        r : 1D array of size 3
            position at which the total intensity is calculated.

        Returns
        -------
        float
            total intensity at the position r.
        '''
        if self.intensity_func != None:
            return self.intensity_func(r)
        else:
            return self.get_intensity_func()(r)
        
    def plot_I_2D(self,ax='x',axshift=0,limits=([-0.05,0.05],[-0.05,0.05])):
        """plot the intensity distribution of all laser beams by using the
        method :func:`get_intensity_func`.
        
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
        """
        axshift = float(axshift)
        xyz = {'x':0,'y':1,'z':2}
        ax_ = xyz[ax]
        del xyz[ax]
        axes_ = np.array([*xyz.values()])
        lim1,lim2 = limits
        x1,x2 = np.linspace(lim1[0],lim1[1],201),np.linspace(lim2[0],lim2[1],201)
        Z = np.zeros((len(x1),len(x2)))
        r = np.zeros(3)
        for i in range(201):
            for j in range(201):
                r[ax_] = axshift
                r[axes_] = x1[i],x2[j]
                Z[i,j] = self.I_tot(r)
        
        X1,X2 = np.meshgrid(x1,x2)
        plt.figure('Intensity distribution of all laser beams at {}={:.2f}mm'.format(
            ax,axshift*1e3))
        plt.contourf(X1*1e3,X2*1e3,Z.T,levels=20)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Intensity $I_{tot}$ in W/m$^2$')
        keys = list(xyz.keys())
        plt.xlabel('position {} in mm'.format(keys[0]))
        plt.ylabel('position {} in mm'.format(keys[1]))
    
    def __delitem__(self,index):
        """delete lasers using del system.lasers[<normal indexing>], or delete all del system.lasers[:]"""
        #delete lasers with del system.lasers[<normal indexing>], or delete all del system.lasers[:]
        del self.entries[index]
        self.intensity_func = None
        
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
    def __init__(self,lamb=860e-9,freq_shift=0,pols='lin',pol_direction=None,
                 P=20e-3,I=None,FWHM=5e-3,w=None,
                 w_cylind=.0,r_cylind_trunc=5e-2,dir_cylind=[1,0,0],
                 freq_Rabi=None,k=[0,0,1],r_k=[0,0,0],beta=0.,phi=0.0):        
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
        pols : str, tuple(str,str), optional
            polarization of the laserbeam. Can be either 'lin', 'sigmap' or
            'sigmam' for linear or circular polarized light of the laser.
            For polarization switching a tuple of two polarizations is needed.
            The default is 'lin'.
        pol_direction : str, optional
            optional addition to the ``pols`` parameter to be considered in the
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
            When the given type of the ``pols`` Parameter is not accepted.
            
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
        if np.all(w):
            self.w = w # old **default** value: (2*(pi*1.5e-3**2))**0.5 --> arbitrary value to compare to old MATLAB rate equations
        elif np.all(FWHM):
            self.FWHM = FWHM
        #___intensity definition or calculation via P and beam widths w & w_cylind:
        #: Rabi frequency in terms of angular frequency 2 pi
        self.freq_Rabi = freq_Rabi
        if np.all(freq_Rabi):
            self.I  = 1.0 #arbitrarily setting initial value for intensity since it is adjusted later during the calculation where the levels are involved.
            self._P = None
        # intensity I is important quantity for calculations instead of the power P.
        elif np.any(I):
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
        if type(pols) == tuple and len(pols) == 2:
            pol1, pol2 = self._test_pol(pols[0]), self._test_pol(pols[1])
            self.pol_switching = True
        elif type(pols) == str:
            pol1, pol2 = self._test_pol(pols), self._test_pol(pols)
            self.pol_switching = False
            if pol_direction == None:
                if pol1 == 'lin':      f_q = np.array([0.,1.,0.]) #q= 0; mF -> mF'= mF
                elif pol1 == 'sigmam': f_q = np.array([0.,0.,1.]) #q=+1; mF -> mF'= mF-1
                elif pol1 == 'sigmap': f_q = np.array([1.,0.,0.]) #q=-1; mF -> mF'= mF+1
            else:
                p = pol_direction
                x = np.array([+1., 0,-1.])/np.sqrt(2)
                y = np.array([+1., 0,+1.])*1j/np.sqrt(2)
                z = np.array([ 0, +1, 0 ])
                if len(p) == 1:
                    if p == 'x':   f_q = x
                    elif p == 'y': f_q = y
                    elif p == 'z': f_q = z
                if len(p) == 2:
                    if pol1 == 'sigmam':
                        a1,a2 = -1., -1j
                    elif pol1 == 'sigmap':
                        a1,a2 = +1., -1j
                    if p == 'xy':   f_q = a1*x + a2*y
                    elif p == 'xz': f_q = a1*z + a2*x
                    elif p == 'yz': f_q = a1*y + a2*z
            self.f_q = np.array([ -f_q[2], +f_q[1], -f_q[0] ]) / np.linalg.norm(f_q)
        else:
            raise Exception("Wrong datatype or length of <pol>: either tuple((2)) or str allowed")
        self.pol1,self.pol2 = pol1,pol2
        
    def _test_pol(self,pol):
        pol_list = ['lin','sigmap','sigmam','x','y','z','xy','xz','yz']
        if pol in pol_list:
            return pol
        else:
            raise Exception("'{}' is not valid, pol can only be '{}','{}', or '{}'".format(pol,*pol_list))
            
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
        #'lamb={:.2e}, I={:.2e}, P={:.2e}, FWHM={:.2e} ,f={:.2e}, pols={}, pol_switching={}'.format(self.lamb,self.I,self.P,self.FWHM,(self.pol1,self.pol2),self.pol_switching)    
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
    @property
    def P(self):
        """calculates the Power of the single beam"""
        if np.all(self._P): return self._P
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