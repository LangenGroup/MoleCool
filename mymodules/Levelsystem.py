# -*- coding: utf-8 -*-
"""
Created on Thu May 14 02:03:38 2020

@author: Felix

This module contains all classes and functions to define a System including all ``Level`` objects.

Example
-------
Below an empty Levelsystem is created which automatically initializes an instance
of the ``Groundstates`` and the ``Excitedstates`` classes. Within these instances
the respective ground states and excited states can be added:
    
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
h = 6.62607e-34#;%2*pi*1.054e-34;
c = 2.998e8

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
        return freq(self.grstates[l],self.exstates[u])
    
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        str(self.grstates)
        str(self.exstates)
        return self.description
    
    @property
    def description(self): return "{:d}+{:d} - Levelsystem".format(self.lNum,self.uNum)
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
        
    def add_grstate(self,nu=0,N=1):
        S,I = .5, .5
        for J in np.arange(abs(N-S),abs(N+S)+1):
            for F in np.arange(abs(J-I),abs(J+I)+1):
                for mF in np.arange(-F,F+1):
                    self.entries.append(Groundstate(nu,N,J,I,S,F,mF,p=(-1)**N)) #I,S ???
                    
    def add_lossstate(self,nu):
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
        self.Gamma = get_Gamma()
        
    def add_exstate(self,nu=0,N=0,J=.5,p=+1):
        S,I = .5, .5
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
        
            