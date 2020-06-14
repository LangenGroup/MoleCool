# -*- coding: utf-8 -*-
"""
Created on Thu May 14 02:03:38 2020

@author: fkogel

v1.4.2

This module contains all classes and functions to define a System including
all ``Level`` objects.

Example
-------
Below an empty Levelsystem is created which automatically initializes an instance
of the :class:`Groundstates` and the :class:`Excitedstates` classes.
Within these instances the respective ground states and excited states can be added:
    
    >>> levels = Levelsystem()
    >>> levels.grstates.add_grstate(nu=0,N=1)
    >>> levels.grstates.add_lossstate(nu=1)
    >>> levels.exstates.add_exstate(nu=0,N=0,J=.5,p=+1)
    
Tip
---
Every object of ``Levelsystem`` or ``Level`` class can be printed to display
all attributes via:
    
    >>> print(levels)
    >>> print(levels.grstates)
    >>> print(levels.exstates[0])
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g
from BaFconstants import freq,get_Gamma

#%%
class Levelsystem:  
    def __init__(self):
        self.grstates = Groundstates()
        self.exstates = Excitedstates()
        
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice)): 
            return self.entries[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.entries[i] for i in index] 
        
    def freq_lu(self,l,u):
        """Returns the angular frequency difference between a ground and an 
        excited state.        

        Parameters
        ----------
        l : int
            index of the ground state in the `self.grstates` list of this class.
        u : int
            index of the excited state in the `self.exstates` list of this class.

        Returns
        -------
        float
            angular frequency.
        """
        return freq(self.grstates[l],self.exstates[u])
    
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        str(self.grstates)
        str(self.exstates)
        return self.description
    
    @property
    def description(self):
        """str: Displays a short description with the number of included laser objects."""
        return "{:d}+{:d} - Levelsystem".format(self.lNum,self.uNum)
    @property
    def entries(self): return [*self.grstates.entries,*self.exstates.entries]   
    @property
    def lNum(self): return self.grstates.lNum
    @property
    def uNum(self): return self.exstates.uNum
    @property
    def N(self): return self.lNum + self.uNum
    

#%%
class Groundstates():
    def __init__(self):
        self.entries = []
        
    def add_grstate(self,nu,N=1,S=.5,I=.5):
        """Adds all hyperfine levels within a certain vibrational and
        rotational state as :class:`Groundstate` objects to the self.entries
        list of this class.

        Parameters
        ----------
        nu : int
            vibrational state of the ground state.
        N : int, optional
            rotational angular momentum of the nuclei. The default is 1.
        S : float, optional
            spin quantum number. The default is 0.5.
        I : float, optional
            nuclear spin quantum number. The default is 0.5.
        
        Returns
        -------
        None.
        """
        for J in np.arange(abs(N-S),abs(N+S)+1):
            for F in np.arange(abs(J-I),abs(J+I)+1):
                for mF in np.arange(-F,F+1):
                    self.entries.append(Groundstate(nu,N,J,I,S,F,mF,p=(-1)**N)) #I,S ???
                    
    def add_lossstate(self,nu):
        """Adds a :class:`Lossstate` object to the self.entries
        list of this class.

        Parameters
        ----------
        nu : int
            all ground state levels with the vibrational quantum number `nu`
            and higher vibrational numbers are represented by a single loss state.

        Returns
        -------
        None.

        """
        self.entries.append(Lossstate(nu))
        
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice)): 
            return self.entries[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.entries[i] for i in index] 
    
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        for i in range(self.lNum):
            st = self.entries[i]
            #if st.name == 'Loss state':
                #print(  'Groundstate  {:2d}: nu={} (Loss state)'.format(i,st.nu))
            #else: print('Groundstate  {:2d}: nu={}, J={}, F={}, mF={:+}, p={:+}'.format(i,st.nu,st.J,st.F,st.mF,st.p))
            print('{:2d} {}'.format(i,st))
        return "{:d} - Groundstates".format(self.lNum)
    
    @property
    def lNum(self): return len(self.entries)

#%%    
class Excitedstates():
    def __init__(self):
        self.entries = []
        #: decay rate :math:`\Gamma` which is received from the function
        #: :func:`BaFconstants.get_Gamma`
        self.Gamma = get_Gamma()
        
    def add_exstate(self,nu,N=0,J=.5,p=+1,S=.5,I=.5):
        """Adds all hyperfine levels within a certain vibrational and
        rotational state as :class:`Excitedstate` objects to the self.entries
        list of this class.

        Parameters
        ----------
        nu : int
            vibrational state of the excited state.
        N : int, optional
            intermediary angular momentum. The default is 0.
        J : float, optional
            Total angular momentum. The default is 0.5.
        p : int, optional
            parity of the excited state. Either +1 or -1. The default is +1.
        S : float, optional
            spin quantum number. The default is 0.5.
        I : float, optional
            nuclear spin quantum number. The default is 0.5.

        Returns
        -------
        None.
        """
        for F in np.arange(abs(J-I),abs(J+I)+1):
            for mF in np.arange(-F,F+1):
                self.entries.append(Excitedstate(nu,N,J,I,S,F,mF,p)) #I,S ???
                
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice)): 
            return self.entries[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.entries[i] for i in index] 
    
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        for i in range(self.uNum):
            st = self.entries[i]
            #print('Excitedstate {:2d}: nu={}, J={}, F={}, mF={:+}, p={:+}'.format(i,st.nu,st.J,st.F,st.mF,st.p))
            print('{:2d} {}'.format(i,st))
        return "{:d} - Excitedstates".format(self.uNum)
    
    @property
    def uNum(self): return len(self.entries)
    
    
#%%
class Level:
    def __init__(self,nu,N,J,I,S,F,mF,p):
        self.nu = nu
        self.N = N
        self.J,self.I,self.S,self.F,self.mF = J,I,S,F,mF
        self.p = p
        self.name = 'Level'
        
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        return '{} : nu={}, N={}, J={}, F={}, mF={:+}, p={:+}'.format(
            self.name, self.nu,self.N,self.J,self.F,self.mF,self.p)
    
class Lossstate(Level):
    def __init__(self,nu):
        self.nu = nu
        self.name, self.key = 'Loss state','l'
    
    def __str__(self):
        return '{} : nu={} (Loss state)'.format('Ground state',self.nu)
        
class Groundstate(Level):
    def __init__(self,nu,N,J,I,S,F,mF,p):
        super().__init__(nu,N,J,I,S,F,mF,p)
        self.name,self.key = 'Ground state','l'
        self.Lambda = 0
        
class Excitedstate(Level):
    def __init__(self,nu,N,J,I,S,F,mF,p):
        super().__init__(nu,N,J,I,S,F,mF,p) #I,S ???
        self.name,self.key = 'Excited state','u'
        self.Lambda = 1
        
            