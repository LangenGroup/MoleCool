# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:34:09 2020

@author: fkogel

v2.2.0

This module contains all classes and functions to define a System including all ``Laser`` objects.

Example
-------
Below an empty Lasersystem is created and a single Laser with wavelength 860nm
and Power 20 mW with linear polarization is added:
    
    >>> lasers = Lasersystem()
    >>> lasers.add(860e-9,20e-3,'lin')

But first start python and import the module::
    
    $ python
    >>> import Lasersystem
    
Tip
---
Every object of :class:`Lasersystem` or :class:`Laser` class can be printed
to display all attributes via:
    
    >>> print(lasers)
    >>> print(lasers[0])
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g

#%%
class Lasersystem:
    """System consisting of :py:class:`Lasersystem.Laser` objects and methods to add them properly.
    
    Example
    -------
    >>> lasers = Lasersystem()
    >>> lasers.add_sidebands(860e-9,20e-3,'lin',-20e6,39e6)
    >>> print(lasers)
    
    
    """
    def __init__(self):
        """an empty `self.entries` list is created to store all :class:`Laser`
        objects.
        """
        self.entries = []
        #: Polarization switching frequency. Default is 5e6.
        self.freq_pol_switch = 5e6 

    def add(self,lamb,P,pols,**kwargs):
        """
        Note
        ----
        Is the same as:
            >>> self.entries.append(Laser(...)).
        
        Parameters
        ----------
        Same as in the ``__init__`` method of the class :class:`Laser` (further information)
        
        Returns
        -------
        None.

        """
        self.entries.append( Laser( lamb, P, pols, **kwargs) ) 
        
    def add_sidebands(self,lamb,P,pols,AOM_shift,EOM_freq,ratios=[0.8,1,1,0.8],**kwargs):
        """Adds four ``Lasers`` as sidebands (shifted by a certain AOM and EOM
        frequencies) instead of a single ``Laser`` object to the ``Lasersystem``.
        
        Parameters
        ----------
        lamb : float
            wavelength of the main transition.
        P : float
            Power of the Laser, i.e. sum of the powers of all sidebands.
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
        **kwargs : TYPE
            optional arguments  (see :class:`Laser`).

        Returns
        -------
        None.

        """
        P_red       = np.array(ratios)/np.sum(ratios) *P
        EOM_freqs   = np.array([-2,-1,1,2])*EOM_freq
        for i in range(4):
            self.entries.append(Laser( lamb, P_red[i], pols,
                freq_shift=AOM_shift+EOM_freqs[i], **kwargs) )
    
    def __delitem__(self,index):
        #delete lasers with del system.lasers[<normal indexing>], or delete all del system.lasers[:]
        del self.entries[index]
        
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice)): 
            return self.entries[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.entries[i] for i in index]
    
    def __str__(self):
        """__str__ method is called when an object of a class is printed with print(obj)"""
        for i in range(self.pNum):
            la = self.entries[i]
            #print('Laserbeam {:2d}: lamb={:.2e}, P={:.2e}, f={:.2e}, pols={}, pol_switching={}'.format(i,la.lamb,la.P,la.f,(la.pol1,la.pol2),la.pol_switching))
            print('Laserbeam {:2d}: {}'.format(i,la))
        return self.description
    @property
    def description(self):
        """str: Displays a short description with the number of included laser objects."""
        return "{:d} - Lasersystem".format(self.pNum)
    @property
    def pNum(self):
        """int: returns the number of included Laser objects."""
        return len(self.entries)
    
#%%
class Laser:
    name = None #cooling / repumping laser
    def __init__(self,lamb,P,pols,pol_direction=None,freq_shift=0,phi=0.0,freq_Rabi=None,
                 FWHM=None,w=None, k=[1,0,0],r_k=[0,0,0],w_cylind=.0,beta=0.):
        """Sets up an Laser with its properties and can be included in the Lasersystem class.
        
        Note
        ----
        freq_shift without 2pi factor
        
        Parameters
        ----------
        lamb : float
            wavelength lambda.
        P : float
            Laser power in W
        pols : str, tuple(str,str)
            polarization of the laserbeam. Can be either 'lin', 'sigmap' or
            'sigmam' for linear or circular polarized light of the laser.
            For polarization switching a tuple of two polarizations is needed.
        freq_shift : float, optional
            Shift of the laserfrequency (without 2 pi) additional to the
            frequency determined by Parameter lamb. The default is 0.0.
        FWHM : float, optional
            FullWidthHalfMaximum of the Gaussian intensity distribution of the
            laserbeam. The default is None.
        w : float, optional
            :math:`1/e^2` beam radius of the Gaussian intensity distribution.
            The default is None.
        w_cylind : float, optional
            :math:`1/e^2` beam radius of the Gaussian intensity distribution
            along x direction for the specific configuration where the
            laser beam is aligned in y axis direction and has a widened intensity
            distribution along x axis with radius `w_cylind`. The distribution
            along the z axis is given by the radius `w`.
            The default is 0.0.
        k : list or array type of dimension 3, optional
            direction of the wave vector of the laserbeam. The inserted array
            is automatically normalized to unit vector. The default is [1,0,0].
        r_k : list or array type of dimension 3, optional
            a certain point which is located anywhere within the laserbeam.
            The default is [0,0,0].
        beta : float, optional
            When the frequency of the laser should be varied linearly in time,
            then `beta` defines the chirping rate in Hz/s (without factor of 2 pi).
            The default is 0.0.

        Raises
        ------
        Exception
            When the given type of the ``pols`` Parameter is not accepted.

        Returns
        -------
        None.
        """
        self.f      = c/lamb + freq_shift
        self.lamb   = c/self.f
        #: absolute value of the wave vector (=2*pi/lambda = omega/c)
        self.kabs   = 2*pi/self.lamb
        # since k is a unit vector
        self.P      = P  
        if FWHM == None:
            if w == None:
                self.w      = (2*(pi*1.5e-3**2))**0.5
            else:
                self.w = w #: :math:`1/e^2` radius of the beam
            self.FWHM = 2*self.w / ( np.sqrt(2)/np.sqrt(np.log(2)) ) 
        else:
            self.FWHM = FWHM
            self.w = np.sqrt(2)/np.sqrt(np.log(2))*FWHM/2 # ~= 1.699*FWHM/2
        
        #: float: :math:`I =P/A` with the Area :math:`A=\pi w_1 w_2/2` of a 2dim Gaussian beam
        self.I      = 2*self.P/(pi*self.w**2) #self.P/(pi*1.5e-3)**2 ??? Laserwaistwidth 
        if w_cylind != .0:
            self.I = self.I*self.w/w_cylind 
        # Currently with **fixed** width of 1.5e-3
        #: float: angular frequency
        self.omega  = 2*pi*self.f
        #: Energy f*h
        self.E      = self.f * h
        """float: Energy of the photons of the laser."""
        self.k      = np.array(k)/np.linalg.norm(k) #unit vector
        self.r_k    = np.array(r_k) #point which is lying in the laserbeam
        self.w_cylind = w_cylind
        self.beta   = beta
        self.phi    = phi
        self.freq_Rabi = freq_Rabi #Rabi frequency in terms of angular frequency 2 pi
        
        if type(pols) == tuple and len(pols) == 2:
            pol1, pol2 = self._test_pol(pols[0]), self._test_pol(pols[1])
            self.pol_switching = True
        elif type(pols) == str:
            pol1, pol2 = self._test_pol(pols), self._test_pol(pols)
            self.pol_switching = False
            if pol_direction == None:
                if pol1 == 'lin':       self.f_q = np.array([0,1.,0]) #q= 0; mF -> mF'= mF
                elif pol1 == 'sigmam':    self.f_q = np.array([1.,0,0]) #q=-1; mF -> mF'= mF-1
                elif pol1 == 'sigmap':    self.f_q = np.array([0,0,1.]) #q=+1; mF -> mF'= mF+1
            else:
                p = pol_direction
                x = np.array([+1., 0,-1.])/np.sqrt(2)
                y = np.array([+1., 0,+1.])*1j/np.sqrt(2)
                z = np.array([ 0, +1, 0 ])
                if len(p) == 1:
                    if p == 'x': f_q = x
                    if p == 'y': f_q = y
                    if p == 'z': f_q = z
                if len(p) == 2:
                    if pol1 == 'sigmap':
                        a1,a2 = -1., -1j
                    elif pol1 == 'sigmam':
                        a1,a2 = +1., -1j
                    if p == 'xy':   f_q = a1*x + a2*y
                    elif p == 'xz': f_q = a1*z + a2*x
                    elif p == 'yz': f_q = a1*y + a2*z
                self.f_q = f_q / np.linalg.norm(f_q)
        else:
            raise Exception("Wrong datatype or length of <pol>: either tuple((2)) or str allowed")
        self.pol1,self.pol2 = pol1,pol2
        
    def _test_pol(self,pol):
        pol_list = ['lin','sigmap','sigmam','x','y','z','xy','xz','yz']
        if pol in pol_list:
            return pol
        else:
            raise Exception("'{}' is not valid, pol can only be '{}','{}', or '{}'".format(pol,*pol_list))
    
        #set_I()
        #self.Is     = pi*c*h*Gamma/(3*lamb**3)
        #self.s      = self.I/self.Is
        
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        return 'lamb={:.2e}, P={:.2e}, f={:.2e}, pols={}, pol_switching={}'.format(
            self.lamb,self.P,self.f,(self.pol1,self.pol2),self.pol_switching)
                   
