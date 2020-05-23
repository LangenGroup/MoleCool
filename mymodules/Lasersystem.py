# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:34:09 2020

@author: Felix

This module contains all classes and functions to define a System including all ``Laser`` objects.

Example
-------
Below an empty Lasersystem is created and a single Laser with wavelength 860nm
and 20 mW with linear polarization is added:
    
    >>> lasers = Lasersystem()
    >>> lasers.add(860e-9,20e-3,'lin')

But first start python and import the module::
    
    $ python
    >>> import Lasersystem
    
Tip
---
Every object of ``Lasersystem`` or ``Laser`` class can be printed to display
all attributes via:
    
    >>> print(lasers)
    >>> print(lasers[0])
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g
h = 6.62607e-34#;%2*pi*1.054e-34;
c = 2.998e8

#%%
class Lasersystem:
    """System consisting of Laser objects and methods to add them properly.
    
    Example
    -------
    >>> lasers = Lasersystem()
    >>> lasers.add_sidebands(860e-9,20e-3,'lin',-20e6,39e6)
    >>> print(lasers)
    
    """
    def __init__(self):
        self.entries = []
        self.freq_pol_switch = 5e6 # Polarization switching frequency

    def add(self,lamb,P,pols,freq_shift=0,w=1e-3):
        """
        Note
        ----
        Is the same as self.entries.append(Laser(...)).
        
        Parameters
        ----------
        Same as in the ``__init__`` method of the class Laser (further information)
        
        Returns
        -------
        None.

        """
        self.entries.append( Laser( lamb, P, pols, freq_shift) ) 
        
    def add_sidebands(self,lamb,P,pols,AOM_shift,EOM_freq):
        EOM_freqs   = np.array([-2,-1,1,2])*EOM_freq
        P_red       = np.array([0.8,1,1,0.8])*P
        for i in range(4):
            self.entries.append(Laser( lamb, P_red[i], pols,
                freq_shift=AOM_shift+EOM_freqs[i]) )
            
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
    def __init__(self,lamb,P,pols,freq_shift=0,w=1e-3):
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
            frequency determined by Param lamb. The default is 0.
        w : str, optional
            laserbeam diameter. The default is 1e-3

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
        self.k      = 2*pi/self.lamb # = omega/c
        self.P      = P
        self.w      = w #: 1/e^2 radius of the beam
        #: float: Currently with **fixed** width of 1.5e-3
        self.I      = self.P/(pi*1.5e-3)**2 #2*self.P/(pi*w**2) ??? Laserwaistwidth 
        self.omega  = 2*pi*self.f
        self.E      = self.f * h
        """float: Energy of the photons of the laser."""
        
        
        if type(pols) == tuple and len(pols) == 2:
            pol1, pol2 = self._test_pol(pols[0]), self._test_pol(pols[1])
            self.pol_switching = True
        elif type(pols) == str:
            pol1, pol2 = self._test_pol(pols), self._test_pol(pols)
            self.pol_switching = False
        else:
            raise Exception("Wrong datatype or length of <pol>: either tuple((2)) or str allowed")
        self.pol1,self.pol2 = pol1,pol2
        
    def _test_pol(self,pol):
        pol_list = ['lin','sigmap','sigmam']
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
            
