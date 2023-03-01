# -*- coding: utf-8 -*-
"""
Created on Thu May 14 02:03:38 2020

@author: fkogel

v3.0.7

This module contains all classes and methods to define all **states** and their
**properties** belonging to a certain Levelsystem.

===============
Class Structure
===============

The general class structure can include several electronic states :class:`ElectronicState`,
i.e. one ground (:class:`ElectronicGrState`) and several excited (:class:`ElectronicExState`)
electronic states. These electronic states in turn include all actual single
quantum states (instances of :class:`State`) with a certain set of quantum numbers.
So, this leads to a well defined basis of each electronic state.

Example
-------

First, let's create empty Levelsystem instance and add e.g. the two electronic
states :math:`5^2S_{1/2}` and :math:`5^2P_{3/2}` as ground and excited states
of 87Rb::
    
    levels = Levelsystem()
    levels.add_electronicstate('S12', 'gs')     # define as ground state ('gs')
    levels.add_electronicstate('P32', 'exs', Gamma=6.065) # decay rate of excited state 6.065 MHz
    levels.S12.add(J=1/2,F=[1,2])
    levels.P32.add(J=3/2,F=[0,1,2,3])
    print(levels) # with this command we get an overview of all states with their quantum numbers.

================
Level Properties
================

The properties of the Levelsystem which are reuqired for simulation the rate equations
or optical Bloch equations are either imported from a .json file or constructed
automatically and can be modified afterwards.
These properties comprise most importantly:
    
    * the electric dipole matrix and branching ratios
    * transition frequenies
    * magnetic g-factors
    * lifetimes of the excited states
    * the mass of the atom or molecule

To modify these properties, simply take the (pandas) DataFrames and change the
values in these frames.

Example
-------

Taking the example above as starting point, we can now change e.g. the wavelength,
the magentic g-factor of one of the ground states, and the ground state hyperfine
splitting.::
    
    levels.wavelengths.iloc[:,:] = 780.241 # in nm
    print(levels.wavelengths)
    
    levels.S12.gfac.iloc[0] = 0.1234
    print(levels.S12.gfac)
    
    levels.S12.freq.iloc[0] = -4.272
    levels.S12.freq.iloc[1] = 2.563
    print(levels.S12.freq)
    
Tip
---
Every object of the classes :class:`Levelsystem` or :class:`State` can be
printed to display all attributes via::
    
    print(levels)       # all electronic states and their specific quantum numbers.
    print(levels.S12)   # all states within S12
    print(levels.S12[0]) # only the first state defined within S12
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g
from scipy.constants import u as u_mass
import constants
from collections.abc import Iterable
import matplotlib.pyplot as plt
import warnings
import os
import numbers
from copy import deepcopy
import pandas as pd
from sympy.physics.wigner import wigner_3j,wigner_6j
import json
#%%
class Levelsystem:
    def __init__(self,load_constants=None,verbose=True):
        """Levelsystem consisting of instances or :py:class:`Levelsystem.ElectronicState`
        and methods to add them properly.
        These respective objects can be retrieved and also deleted by using the
        normal item indexing of a :class:`Levelsystem`'s object::
            
            levels = Levelsystem()
            levels.add_electronicstate('S12', 'gs')     # define as ground state ('gs')
            levels.S12.add(J=1/2,F=[1,2])
            state1 = levels.S12[0]
            print(state1)
            del levels.S12[-1] # delete last added State instance 
            del levels['S12']  # delete complete electronic state

        Parameters
        ----------
        load_constants : str, optional
            the constants of the levelsystem can be imported from an .json file.
            If this is desired provide the respective filename without the .json
            extension. The default is None.
        verbose : bool, optional
            Specifies if additional warnings should be printed during the
            level construction. The default is True.
        
        Tip
        ---
        When arbitrary custom level systems want to be defined, first all
        levels have to be added.
        
        Then the default constants and properties can be nicely viewed with
        the function :func:`print_properties`. Afterwards the values in these
        pandas.DataFrames (here: vibrational branchings, transition wavelength,
        and g-factor) can be easily modified via `<DataFrame>.iloc[<index>]`.
        
        Tip
        ---
        Important properties within an instance :class:`Levelsystem` can be
        accessed with the ´get_<property>()´ methods or directly via the properties
        without ´get_´ in their names, like:
            
            * dMat          # electric dipole matrix
            * dMat_red      # reduced electric dipole matrix
            * vibrbranch    # vibrational branching ratios
            * wavelengths   # wavelengths (in nm) between the transitions
        """
        #: list containing the labels of the ground electronic states
        self.grstates_labels   = [] #empty lists to hold labels of electronic states
        #: list containing the labels of the excited electronic states
        self.exstates_labels   = [] #to be appended later
        self.verbose                    = verbose
        # loading the dictionary with the constants from json file
        self.load_constants             = load_constants
        self.const_dict                 = get_constants_dict(load_constants)
        # initialize all property instances
        self.reset_properties()
        
    def __getitem__(self,index):
        if self.N == 0: raise Exception('No states are defined/ included within this Electronic State!')
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice)): 
            return self.states[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.states[i] for i in index]
    
    def __delitem__(self,label):
        """delete electronic states del system.levels[<electronic state label>], or delete all del system.levels[:]"""
        if type(label) != str:
            raise ValueError('index for an electronic state to be deleted must be provided as string of the label')
        if label in self.grstates_labels:
            self.grstates_labels.remove(label)
        else:
            self.exstates_labels.remove(label)
        del self.__dict__[label]
        self.reset_properties()
        print('ElectronicState {} deleted and all properties resetted!'.format(label))
        
    def add_all_levels(self,v_max):
        """Function to add all ground and excited states of a molecule with a
        loss state in a convenient manner.

        Parameters
        ----------
        v_max : int
            all ground states with vibrational levels :math:`\\nu\\le` `v_max`
            and respectively all excited states up to `(v_max-1)` are added to
            the subclasses :class:`Groundstates` and :class:`Excitedstates`.
        """
        for key in list(self.const_dict['level-specific'].keys()):
            gs_exs,label = constants.split_key(key)
            self.add_electronicstate(label, gs_exs)
            
            if gs_exs == 'gs':
                self.__dict__[label].load_states(v=np.arange(0,v_max+.5,dtype=int))
                self.__dict__[label].add_lossstate(v=v_max+1)
            elif gs_exs == 'exs':
                if v_max == 0: self.__dict__[label].load_states(v=0)
                else:
                    self.__dict__[label].load_states(v=np.arange(0,v_max-.5,dtype=int))
        
    def add_electronicstate(self,label,gs_exs,load_constants=None,**kwargs):
        """adds an electronic state (ground or excited state) as instance of
        the class :class:`ElectronicState` to this :class:`~Molecule.Molecule`.

        Parameters
        ----------
        label : str
            label or name of the electronic state so that this electronic state
            will be accessible via levels.<label>.
        gs_exs : str
            determines whether an electronic ground or excited state should be
            added. Therefore, `gs_exs` can be either 'gs' or 'exs'.
        load_constants : str, optional
            the constants of the levelsystem can be imported from an .json file.
            If this is desired provide the respective filename without the .json
            extension. The default is None.
        **kwargs : kwargs
            keyword arguments for the eletronic state (see
            :class:`ElectronicState` for the specific parameters)
        """
        if not label.isidentifier():
            raise ValueError('Please provide a valid variable/ instance name for `label`!')
        if (not load_constants) and self.load_constants:
            load_constants = self.load_constants
        if not ('verbose' in kwargs):
            kwargs['verbose'] = self.verbose
        if label in [*self.grstates_labels,*self.exstates_labels]:
            raise Exception('There is already an electronic state {} defined!'.format(label))
        if gs_exs == 'gs':
            if len(self.grstates_labels) == 1:
                raise Exception('There is already one electronic ground state defined!')
            self.__dict__[label] = ElectronicGrState(load_constants=load_constants,label=label,**kwargs)
            self.grstates_labels.append(label)
        elif gs_exs == 'exs':
            self.__dict__[label] = ElectronicExState(load_constants=load_constants,label=label,**kwargs)
            self.exstates_labels.append(label)
        else:
            raise ValueError("Please provide 'gs' or 'exs' as `gs_exs` for the electronic ground or excited states")
        
    #%% get functions
    def get_dMat(self,gs=None,exs=None):
        """Function which returns the electric dipole matrix. This matrix is
        either simply loaded from the :py:class:`constants` module or constructed
        with the reduced electric dipole matrix given by the function
        :func:`get_dMat_red`.
        

        Parameters
        ----------
        calculated_by : str, optional
            Additional parameter if multiple different matrices are available
            for one system. The default is 'YanGroupnew'.

        Returns
        -------
        pandas.DataFrame
            Electric dipole matrix.
        """
        #this function has to be resetted after a change of dMat_from which the new dMat should be calculated
        if gs == None:  gs =    self.grstates_labels[0]
        if exs == None: exs =   self.exstates_labels[0]
        
        if gs in self._dMat:
            if exs in self._dMat[gs]:
                if len(self._dMat[gs][exs]) != 0:
                    return self._dMat[gs][exs]
        #gs_exs_label = '-'.join([gs,exs]) #df1.index.to_frame()['gs'].iloc[0]#ß? #df1.index.names.index('gs') #df1.columns.to_frame()['exs'].iloc[0]
        DF_dMat         = constants.get_DataFrame(self.const_dict,'dMat',gs=gs,exs=exs)
        DF_dMat_red     = constants.get_DataFrame(self.const_dict,'dMat_red',gs=gs,exs=exs)
        DF_branratios  = constants.get_DataFrame(self.const_dict,'branratios',gs=gs,exs=exs)
        if len(DF_dMat) != 0:
            dMat = DF_dMat
        elif (len(DF_dMat_red) == 0) and (len(DF_branratios) !=0):
            self._branratios = DF_branratios
            if self.verbose: warnings.warn('No dipole matrix or reduced dipole matrix found in constants.py, so the dipole matrix is constructed from the given branching ratios only with positive values!')
            dMat = self._branratios**0.5
        else:
            dMat_red = self.get_dMat_red(gs=gs,exs=exs)
            dMat = []
            index = []
            for index1,row1 in dMat_red.iterrows():
                F = index1[dMat_red.index.names.index('F')]
                for mF in np.arange(-F,F+1):
                    dMat_row = []
                    index.append([*index1,mF])
                    columns = []
                    for index2,row2 in row1.iteritems():
                        F_ = index2[row1.index.names.index('F')]
                        for mF_ in np.arange(-F_,F_+1):
                            dMat_row.append( row2 * (-1)**(F_-mF_) * float(wigner_3j(F_,1,F,-mF_,mF_-mF,mF)) )
                            columns.append([*index2,mF_])
                    dMat.append(dMat_row)
            dMat = pd.DataFrame(dMat,
                                index  =pd.MultiIndex.from_arrays(np.array(index,dtype=object).T,   names=(*dMat_red.index.names,'mF')),
                                columns=pd.MultiIndex.from_arrays(np.array(columns,dtype=object).T, names=(*dMat_red.columns.names,'mF'))
                                     )
            dMat /= np.sqrt((dMat**2).sum(axis=0))
        
        if gs in self._dMat:
            self._dMat[gs][exs] = dMat    
        else:
            self._dMat[gs] = {exs: dMat}
        return dMat
    
    def get_dMat_red(self,gs=None,exs=None):
        """Function which returns the reduced electric dipole matrix. This matrix is
        either simply loaded from the :py:class:`constants` module or constructed
        with the electric dipole matrix given by the function :func:`dMat_red`. If no
        dipole matrix or a reduced one is available a new reduced matrix is
        constructed with only ones for each transition which can be adjusted
        afterwards.
        
        Returns
        -------
        pandas.DataFrame
            Reduced electric dipole matrix.
        """
        if gs == None:  gs =    self.grstates_labels[0]
        if exs == None: exs =   self.exstates_labels[0]
        
        if gs in self._dMat_red:
            if exs in self._dMat_red[gs]:
                if len(self._dMat_red[gs][exs]) != 0:
                    return self._dMat_red[gs][exs]
                
        DF_dMat_red = constants.get_DataFrame(self.const_dict,'dMat_red',gs=gs,exs=exs)
        if len(DF_dMat_red) != 0:
            dMat_red = DF_dMat_red
            # must be updated below!
        elif len(constants.get_DataFrame(self.const_dict,'dMat',gs=gs,exs=exs)) != 0:
            dMat = self.get_dMat(gs=gs,exs=exs).sort_index(axis='index')
            dMat_red = dMat.copy() #,level=[0,1])
            for J,F,mF in dMat_red.index:
                for J_,F_,mF_ in dMat_red.columns:
                    grtoex = True
                    if grtoex:
                        Fa,Ma,Fb,Mb = F_,mF_,F,mF
                    else: #extogr == True:
                        Fa,Ma,Fb,Mb = F,mF,F_,mF_
                    factor = (-1)**(Fa-Ma)*np.sqrt(2*Fa+1) * float(wigner_3j(Fa,1,Fb,-Ma,Ma-Mb,Mb))
                    if factor != 0.0:
                        dMat_red.loc[(J,F,mF),(J_,F_,mF_)] /= factor
            return dMat_red
        else:
            # self._dMat[gs][exs] = None #remove this?
            dMat_red = pd.DataFrame(1.0,index=self.__dict__[gs].DFzeros_without_mF().index,
                                    columns=self.__dict__[exs].DFzeros_without_mF().index)
            if self.verbose:
                warn_txt = 'There is no dipole matrix or reduced dipole matrix available!' + \
                    'So a reduced matrix has been created only with ones:\n{}'.format(dMat_red)
                warnings.warn(warn_txt)
                
        if gs in self._dMat_red:
            self._dMat_red[gs][exs] = dMat_red    
        else:
            self._dMat_red[gs] = {exs: dMat_red}
        return dMat_red

    def get_vibrbranch(self,gs=None,exs=None):
        """Function returns a matrix for the vibrational branching ratios between
        vibrational excited levels with :math:`\\nu` and ground levels wth
        :math:`\\nu'`.This matrix is either simply loaded from the
        :py:class:`constants` module or constructed with the same branching ratios
        for all transitions.        

        Returns
        -------
        pandas.DataFrame
            vibrational branching ratios matrix.
        """
        if gs == None:  gs =    self.grstates_labels[0]
        if exs == None: exs =   self.exstates_labels[0]
        
        if gs in self._vibrbranch:
            if exs in self._vibrbranch[gs]:
                if len(self._vibrbranch[gs][exs]) != 0:
                    return self._vibrbranch[gs][exs]
        
        DF_vibrbranch = constants.get_DataFrame(self.const_dict,'vibrbranch',gs=gs,exs=exs)
        if len(DF_vibrbranch) != 0:
            vibrbranch = DF_vibrbranch
        else: #old and simpler version:
            # vibrbranch = pd.DataFrame(np.ones((self.__dict__[gs].v_max+1,self.__dict__[exs].v_max+1))/(self.__dict__[gs].v_max+1))
            # vibrbranch.rename_axis("v",axis=0,inplace=True)
            # vibrbranch.rename_axis("v'",axis=1,inplace=True)    
            DF_gs = self.__dict__[gs].DFzeros_without_mF()
            DF_exs = self.__dict__[exs].DFzeros_without_mF()
            vibrbranch = pd.DataFrame(1.0,index=DF_gs.index.droplevel([QuNr for QuNr in DF_gs.index.names if QuNr not in ['gs','exs','v']]).drop_duplicates(),
                                    columns=DF_exs.index.droplevel([QuNr for QuNr in DF_exs.index.names if QuNr not in ['gs','exs','v']]).drop_duplicates())
            vibrbranch /= vibrbranch.sum(axis=0)
        if gs in self._vibrbranch:
            self._vibrbranch[gs][exs] = vibrbranch
        else:
            self._vibrbranch[gs] = {exs: vibrbranch}
        
        return vibrbranch
        
    def get_wavelengths(self,gs=None,exs=None):
        """Function returns a list of matrices for nicely displaying
        the wavelengths between the vibrational transitions and the 
        frequencies between hyperfine transitions to conveniently specifying
        or modifying all participating transitions. These wavelengths and
        frequencies are loaded from the :py:class:`constants` module if
        available. Otherwise all wavelengths are set to 860e-9 and all other
        hyperfine frequencies to zero to be adjusted.

        Returns
        -------
        list of pandas.DataFrame and pandas.Series entries.
            list of matrices specifying the frequencies of the participating
            transitions.
        """
        if gs == None:  gs =    self.grstates_labels[0]
        if exs == None: exs =   self.exstates_labels[0]
        
        if gs in self._wavelengths:
            if exs in self._wavelengths[gs]:
                if len(self._wavelengths[gs][exs]) != 0:
                    return self._wavelengths[gs][exs]
                
        DF_wavelengths = constants.get_DataFrame(self.const_dict,'vibrfreq',gs=gs,exs=exs)
        if len(DF_wavelengths) != 0:
            wavelengths = DF_wavelengths
        else:
            wavelengths = pd.DataFrame(860.0,index=self.__dict__[gs].DFzeros_without_mF().index,
                                    columns=self.__dict__[exs].DFzeros_without_mF().index)
            
        if gs in self._wavelengths:
            self._wavelengths[gs][exs] = wavelengths
        else:
            self._wavelengths[gs] = {exs: wavelengths}     
        return wavelengths
    
    #: electric dipole matrix
    dMat        = property(get_dMat)
    #: reduced electric dipole matrix
    dMat_red    = property(get_dMat_red)
    #: vibrational branching ratios
    vibrbranch  = property(get_vibrbranch)
    #: transition wavelengths and frequencies
    wavelengths    = property(get_wavelengths)
    #%% calc functions for the rate and optical Bloch equations
    def calc_dMat(self):
        """In contrast to the other functions :func:`get_dMat` or :func:`get_dMat_red`
        this function calculates a the normalized electric dipole matrix as numpy 
        array ready to be directly called and used for the functions
        :func:`~System.System.calc_rateeqs` and :func:`~System.System.calc_OBEs`.
        This matrix includes also the vibrational branching ratios and handles
        the loss state in a correct way and is not meant to be modified.

        Returns
        -------
        numpy.ndarray
            fully normalized electric dipole matrix.
        """
        
        if np.all(self.__dMat_arr) != None: return self.__dMat_arr
        self.__dMat_arr = np.zeros((self.lNum,self.uNum,3))
        
        #levels._dMat.xs((1.5,2,-1),level=('J','F','mF'),axis=0,drop_level=True).xs((0.5,1,-1),level=("J'","F'","mF'"),axis=1,drop_level=True)
        N_grstates = 0
        for Grs_lab, Grs in zip(self.grstates_labels, self.grstates):
            N_exstates = 0
            for Exs_lab,Exs in zip(self.exstates_labels, self.exstates):
                DF_vb   = self.get_vibrbranch(gs=Grs_lab, exs=Exs_lab)
                if 'v' in Grs[0].QuNrs:
                    DF_vb   = DF_vb.iloc[np.argwhere(DF_vb.index.get_level_values('v') < Grs.v_max+0.1)[:,0]]
                DF_vb  /= DF_vb.sum(axis=0) # normalized DF_vb must be multiplied afterwards with a factor due to transition dipole moment
                
                DF_dMat = self.get_dMat(gs=Grs_lab, exs=Exs_lab)
                DF_dMat /= np.sqrt((DF_dMat**2).sum(axis=0))
                for l,gr in enumerate(Grs.states):
                    for u,ex in enumerate(Exs.states):
                        val_vb = self.states_in_DF(gr,ex,DF_vb)
                        if gr.is_lossstate:
                            #the q=+-1,0 entries squared of the dMat are summed in the last line of the equations set in the Fokker-Planck paper.
                            # So for the loss state which should not interact with other levels, it doesn't matter which q component of the sum in the last line is contributing.
                            self.__dMat_arr[N_grstates+l, N_exstates + u, :] = np.array([1,0,0])*np.sqrt(val_vb)
                        else:
                            val_dMat = self.states_in_DF(gr,ex,DF_dMat)
                            pol = ex.mF-gr.mF
                            if (abs(pol) <= 1) and (val_vb != None):
                                self.__dMat_arr[N_grstates+l, N_exstates+u, int(pol)+1] = val_dMat*np.sqrt(val_vb)
                N_exstates += Exs.N
            N_grstates += Grs.N
        
        self.__dMat_arr /= np.sqrt((self.__dMat_arr**2).sum(axis=(2,0)))[None,:,None]
        # np.nan_to_num(self.__dMat_arr,copy=False)
        return self.__dMat_arr
    
    def calc_branratios(self):
        """Function calculates fully normalized branching ratios using the dipole
        matrix calculated in the function :func:`~System.System.calc_dMat`
        (see for more details).
        
        Returns
        -------
        numpy.ndarray
            fully normalized branching ratios.
        """
        if np.all(self.__branratios_arr) != None: return self.__branratios_arr
        self.__branratios_arr = (self.calc_dMat()**2).sum(axis=2)
        return self.__branratios_arr
    
    def calc_freq(self):
        """Function calculates the angular absolute frequency **differences**
        between **all** levels included in this class using the wavelengths and
        frequencies specified by the function :func:`get_freq`. These values are 
        returned as numpy array ready to be directly called and used for the functions
        :func:`~System.System.calc_rateeqs` and :func:`~System.System.calc_OBEs`.
        
        Returns
        -------
        numpy.ndarray
            angular frequency array.
        """
        if np.all(self.__freq_arr) != None: return self.__freq_arr
        self.__freq_arr = np.zeros((self.lNum,self.uNum))
        
        N_grstates = 0
        for Grs_lab, Grs in zip(self.grstates_labels, self.grstates):
            N_exstates = 0
            for Exs_lab,Exs in zip(self.exstates_labels, self.exstates):
                DF = self.get_wavelengths(gs=Grs_lab, exs=Exs_lab)
                for l,gr in enumerate(Grs.states):
                    for u,ex in enumerate(Exs.states):
                        val = self.states_in_DF(gr,ex,DF)
                        if (val != None) and (val != 0):
                            self.__freq_arr[N_grstates+l,N_exstates+u] = c/(val*1e-9)
                N_exstates += Exs.N
            N_grstates += Grs.N
        
        for i0, ElSts in enumerate([self.grstates,self.exstates]):
            N_ElSts = 0
            for ElSt in ElSts:
                DF = ElSt.get_freq()
                for i_st,st in enumerate(ElSt.states):
                    if st.is_lossstate: #leave this restriction here?!
                        continue
                    val = self.state_in_DF(st,DF)
                    if val != None:
                        if i0 == 0:   self.__freq_arr[N_ElSts+i_st,:] -= val*1e6
                        elif i0 == 1: self.__freq_arr[:,N_ElSts+i_st] += val*1e6
                N_ElSts += ElSt.N
            
        self.__freq_arr *= 2*pi #make angular frequencies
        return self.__freq_arr
    
    def states_in_DF(self,st1,st2,DF): #move this and the other function outside of the class with an kwarg verbose.
        ind_names = list(DF.index.names)
        col_names = list(DF.keys().names)
        for index1,row1 in DF.iterrows():
            if DF.shape[0] == 1: index1 = (index1,) #index1 must be a iterable tuple even if it containts only 1 element
            for index2,row2 in row1.iteritems():
                if DF.shape[1] == 1: index2 = (index2,)
                allTrue = [True]
                for i,ind_name in enumerate(ind_names):
                    if not np.all(allTrue): break
                    allTrue.append(index1[i] == st1.__dict__.get(ind_name,None))
                
                for j,col_name in enumerate(col_names):
                    if not np.all(allTrue): break
                    allTrue.append(index2[j] == st2.__dict__.get(col_name,None))
                
                if np.all(allTrue):
                    return row2
        if st1.is_lossstate or st2.is_lossstate:
            return None
        elif (('v' in st1.QuNrs) and (st1.v > 0)) or (('v' in st2.QuNrs) and (st2.v > 0)):
            st1_,   st2_    = st1.copy(), st2.copy()
            st1_.v, st2_.v  = 0, 0
            return self.states_in_DF(st1_,st2_,DF)
        elif self.verbose:
            warnings.warn('No value in DF found for {}, {}'.format(st1,st2))
        return None
    
    def state_in_DF(self,st,DF):
        ind_names = list(DF.index.names)
        for index1,row1 in DF.iteritems():    
            allTrue = [True]
            for i,ind_name in enumerate(ind_names):
                if not np.all(allTrue): break
                allTrue.append(index1[i] == st.__dict__.get(ind_name,None))
            if np.all(allTrue):
                return row1
            
        if ('v' in st.QuNrs) and (st.v > 0) and (not st.is_lossstate):
            st_     = st.copy()
            st_.v   = 0
            return self.state_in_DF(st_,DF)
        elif self.verbose:
            warnings.warn('No value in DF found for {}'.format(st))
        return None
        
    
    def calc_muMat(self):
        """Function calculates the magnetic dipole moment operator matrix for **all** levels
        included in this class using the g-factors specified by the function
        :func:`get_gfac`. These values are returned as numpy array ready to be
        directly called and used for the function :func:`~System.System.calc_OBEs`.
        
        Returns
        -------
        tuple of numpy.ndarray
            magnetic moment operator matrix.
        """    
        if self._muMat != None: return self._muMat
        # mu Matrix for magnetic remixing:
        # this matrix includes so far also off-diagonal non-zero elements (respective to F,F')
        # which will not be used in the OBEs calculation
        self._muMat  = (np.zeros((self.lNum,self.lNum,3)),
                        np.zeros((self.uNum,self.uNum,3)))
        for i0, ElSts in enumerate([self.grstates,self.exstates]):
            N = 0
            for ElSt in ElSts:
                DF = ElSt.get_gfac()
                for i1, st1 in enumerate(ElSt.states):
                    if st1.is_lossstate:
                        continue # all elements self._muMat[i0][i1,N:ElSt.N,:] remain zero
                    val = self.state_in_DF(st1,DF)
                    F,m = st1.F, st1.mF
                    for i2, st2 in enumerate(ElSt.states):
                        if st2.is_lossstate:
                            continue # all elements self._muMat[i0][i1,i2,:] remain zero
                        n = st2.mF
                        for q in [-1,0,1]:
                            if val != None:
                                self._muMat[i0][N+i1,N+i2,q+1] = -val* (-1)**(F-m)* \
                                    np.sqrt(F*(F+1)*(2*F+1)) * float(wigner_3j(F,1,F,-m,q,n))
                N += ElSt.N
        return self._muMat
    
    def calc_M_indices(self):
        """Function returns the indices determining all hyperfine states within
        a specific F or F' of the ground or excited state. These values are
        used in the function :func:`~System.System.calc_OBEs` when looping
        through all hyperfine states in conjunction with the g-factors
        for calculation the effect of a magnetic field.

        Returns
        -------
        tuple of tuple of lists
            indices of the magnetic sublevels belonging to a certain F or F'
            of the ground or excited state.
        """
        if self._M_indices != None: return self._M_indices
        
        M_indices = [[],[]]
        for i0, ElSts in enumerate([self.grstates,self.exstates]):
            N = 0
            for ElSt in ElSts:
                states_list = [st for st in ElSt.states]
                for l1,st1 in enumerate(states_list):
                    list_M = []
                    if st1.is_lossstate:
                        M_indices[i0].append(np.array([N+l1]))
                        continue
                    for l2,st2 in enumerate(states_list):
                        if st2.is_lossstate:
                            continue
                        if st1.is_equal_without_mF(st2):
                            list_M.append(N+l2)
                    M_indices[i0].append(np.array(list_M))
                N += ElSt.N
        
        self._M_indices = (tuple(M_indices[0]),tuple(M_indices[1]))
        return self._M_indices
    
    def calc_Gamma(self):
        """Method calculates the natural decay rate Gamma (in angular frequency)
        for each single excited state.

        Returns
        -------
        np.ndarray
            Gamma array of length uNum as angular frequency [Hz].
        """
        if np.all(self._Gamma) != None:
            return self._Gamma
        else:
            self._Gamma = np.array([ElSt.Gamma for ElSt in self.exstates
                                    for i in range(ElSt.N)]) * 2*pi * 1e6
            return self._Gamma
        
    def calc_all(self):
        # initially calculate every property of the level system so that these
        # arrays can simply be called without calculating them every time:
        self.calc_dMat()
        self.calc_branratios()
        self.calc_freq()
        self.calc_muMat()
        self.calc_M_indices()
        self.calc_Gamma()
        
        # mark that properties are calculated for not adding/deleting more states
        for ElSt in self.electronic_states:
            ElSt.properties_not_calculated = False
    #%%
    def __str__(self):
        """__str__ method is called when an object of a class is printed with print(obj)"""
        for ElSt in self.electronic_states:
            print(ElSt)
        return self.description
    
    def print_properties(self): 
        """Prints all relevant constants and properties of the composed levelsystem
        in a convenient way to modify them if needed afterwards.
        """
        n=40
        print('{s:{c}^{n}}'.format(s='',n=n,c='*'))
        print('{s:{c}^{n}}'.format(s=' Levelsystem ',n=n,c='*'))
        print('{s:{c}^{n}}'.format(s='',n=n,c='*'))
        
        print('\nmass (in u):', self.mass/u_mass)
        print('\n{s:{c}^{n}}'.format(s=' level-specific ',n=n,c='+'))
        for ElSt in self.electronic_states:
            ElSt.print_properties()
        
        print('\n{s:{c}^{n}}'.format(s=' transition-specific ',n=n,c='+'))
        for Grs_lab, Grs in zip(self.grstates_labels, self.grstates):
            for Exs_lab,Exs in zip(self.exstates_labels, self.exstates):
                print('\n{s:{c}^{n}}'.format(s=Grs_lab + ' <- ' + Exs_lab,n=n,c='-'))
                print('\ndipole matrix:',                 self.get_dMat(gs=Grs_lab,exs=Exs_lab),
                      '\nvibrational branching:',         self.get_vibrbranch(gs=Grs_lab,exs=Exs_lab),
                      '\nwavelengths (in nm):',           self.get_wavelengths(gs=Grs_lab,exs=Exs_lab),
                      sep='\n')
        
    def check_config(self,raise_Error=False):
        """checking the configuration of the Levelsystem to be used in calculating
        laser cooling dynamics. E.g. involves to check whether the states are 
        correctly defined.
        
        Parameters
        ----------
        raise_Error : bool, optional
            If the configuration is not perfect, this method raises an error message
            or only prints a warning depending on `raise_Error`. The default is False.
        """
        if len(self.grstates_labels) > 1:
            Err_str = 'Only one electronic ground state can be defined' + \
                '({} are given)'.format(len(self.grstates_labels))
            if raise_Error: raise Exception(Err_str)
            else: warnings.warn(Err_str)
            
        Err_str = 'Electronic state {} contains no levels!'
        for ElSt in self.electronic_states:
            if ElSt.N == 0:
                if raise_Error: raise Exception(Err_str.format(ElSt.label))
                else: warnings.warn(Err_str.format(ElSt.label))
                
    def reset_properties(self):
        """Method resets and initializes all property objects (e.g. dMat, dMat_red,
        vibrbranch, wavelengths, muMat, Mindices, Gamma).
        """
        self.mass                       = self.const_dict.get('mass',0.0)*u_mass #if no value is defined, mass will be set to 0
        # The following variables with one underscore are meant to not directly accessed
        # outside of this class. They can be called and and their values can be modified
        # by using the respective methods "get_<variable>"
        self._dMat                      = {} #dict with first key as gs label and second key (of the nested dict) as exs label
        self._dMat_red                  = {}
        self._vibrbranch                = {}
        self._wavelengths               = {}
        # The following variables with two underscores are only for internal use inside the class
        self.__dMat_arr                 = None
        self.__branratios_arr           = None
        self.__freq_arr                 = None
        self._muMat                     = None
        self._M_indices                 = None
        self._Gamma                     = None

    @property
    def description(self):
        """str: Displays a short description with the number of included state objects."""
        return "{:d}+{:d} - Levelsystem".format(self.lNum,self.uNum)
    
    @property
    def states(self):
        """Returns a combined list of all state objects defined in the individual
        electronic states.
        """
        return [st for ElSt in self.electronic_states for st in ElSt.states]
    
    @property
    def lNum(self):
        '''Returns the total number of states defined in the ground electronic
        states as an integer.'''
        return sum([ElSt.N for ElSt in self.grstates])
    
    @property
    def uNum(self):
        '''Returns the total number of states defined in the excited electronic
        states as an integer.'''
        return sum([ElSt.N for ElSt in self.exstates])
    
    @property
    def N(self):
        '''Returns the total number of states defined in all electronic
        states as an integer, i.e. N = :func:`lNum` + :func:`uNum`.'''
        return self.lNum + self.uNum
    
    @property
    def grstates(self):
        '''Returns a list containing all defined instances of ground electronic
        states (:class:`ElectronicGrState`).'''
        return [self.__dict__[ElSt] for ElSt in self.grstates_labels]
    
    @property
    def exstates(self):
        '''Returns a list containing all defined instances of excited electronic
        states (:class:`ElectronicExState`).'''
        return [self.__dict__[ElSt] for ElSt in self.exstates_labels]
    
    @property
    def electronic_states(self):
        '''Returns a list containing all defined instances of electronic
        states, i.e. stacking list of :func:`grstates` and :func:`exstates`.'''
        return [*self.grstates,*self.exstates]
   
#%% #########################################################################
class ElectronicState():
    def __init__(self,label='',load_constants=None,verbose=True):
        #: list for storing the pure states which can be added after class initialization
        self.states = []
        #determine the class instance's name from label
        # X,A,B,.. if state is electronic ground/ excited state
        
        self.gs_exs = ''
        self.label  = label
        self.verbose = verbose
        # load_constants parameter can be specified but by default it is automatically
        # imported from the same variable in the Levelsystem class
        self.load_constants     = load_constants
        self.const_dict         = get_constants_dict(load_constants)
        self.properties_not_calculated = True
        self._freq = []
        self._gfac = []

    def add(self,**QuNrs):
        """Method adds an instance of :class:`State` to this electronic state.
        
        Using this method arbitrary quantum states with their respective quantum
        numbers can be added to construct a certain levelsystem.
        Calculating all properties of the Levelsystem class does only work if all
        levels are added first, and then the calculations are done afterwards.
        
        Parameters
        ----------
        **QuNrs : kwargs of int or float
            Quantum numbers of the state, e.g. J=1/2, F=2. Providing the quantum
            number F is mandatory however.
            
        Note
        ----
        The Quantum numbers can be arbitrarily provided (e.g. J,S,N,I,p,..).
        However, there are requirements for the following quantum numbers:
            
        F : float or iterable
            Total angular momentum typically including the nuclear spin.
            **This quantum number F must be given.**
        v : int, optional
            vibrational state manifold quantum number. This quantum number is
            mandatory if one want to simulate branchings without selection rules,
            like for the vibrational states in molecules.
        mF : float, optional
            magnetic sublevel quantum number. If it is not provided, then all
            possible magnetic sublevels (mF=-F,-F+1,...,F) will be added automatically.
            The absolute value of mF must fulfill the relation :math:`|m_F| \le F`.
        """
        def isnumber(x):
            return isinstance(x,numbers.Number)
        
        if not ('F' in QuNrs):
            raise KeyError("Key `F` is not provided !")
        F = QuNrs['F']
        
        if 'mF' in QuNrs:
            mF = QuNrs['mF']
            if mF > F:
                raise ValueError('The absolute value of mF must be equal or lower than F')
        else:
            mF = None
            
        if isnumber(F):
            if isinstance(mF,(list,np.ndarray)):
                pass
            elif mF == None:
                mF = np.arange(-F,F+1)
            elif isnumber(mF):
                mF = [mF]
            else: raise Exception('Wrong datatype of mF')
            for mF_i in mF:
                QuNrs['mF'] = mF_i
                if self.gs_exs not in QuNrs:
                    QuNrs = {self.gs_exs:self.label,**QuNrs}
                state = State(is_lossstate=False,**QuNrs)
                if self.state_exists(state):
                    raise Exception('{} already exists!'.format(state))
                if self.properties_not_calculated:
                    self.states.append(state)
                else:
                    raise Exception('After any property is initialized, one can not add more states')
                    
        elif isinstance(F, Iterable):
            for F_i in F:
                self.add(**{**QuNrs,'F':F_i})
        else:
            raise ValueError('Wrong datatype of parameter F')
                
    def state_exists(self,state):
        #test if a given state already exists in the ElectronicState and returns boolean
        for st in self.states:
            if st == state:
                return True
        return False
    
    def load_states(self,**QuNrs):
        if len(self.const_dict) == 0:
            raise Exception('There is no constants dictionary available to load states from!')
        if isinstance(QuNrs.get('v',None),Iterable):
            for v_i in QuNrs['v']:
                QuNrs['v'] = v_i
                self.load_states(**QuNrs) #recursively calling this method with no 'v' Iterable
        else:
            list_of_dicts = constants.get_levels(dic=self.const_dict,gs_exs=self.label,**QuNrs)
            if not list_of_dicts:
                text = 'No pre-defined states found for electronic state {} with: {}'.format(
                    self.label, ', '.join(['{}={}'.format(key,QuNrs[key]) for key in QuNrs]))
                if 'v' in QuNrs:
                    # set vibrational quantum number to 0 and try to import constants
                    v_notfound = QuNrs['v']
                    QuNrs['v'] = 0
                    list2_of_dicts = constants.get_levels(dic=self.const_dict,gs_exs=self.label,**QuNrs)
                    if list2_of_dicts:
                        for dict_QuNrs in list2_of_dicts:
                            dict_QuNrs_v = {**dict_QuNrs}
                            dict_QuNrs_v['v'] = v_notfound
                            self.add(**dict_QuNrs_v)
                    if self.verbose:
                        warnings.warn(text+'\n...instead the same states as for v=0 were imported!')
                else:
                    if self.verbose:
                        warnings.warn(text)
            else:
                for dict_QuNrs in list_of_dicts:
                    self.add(**dict_QuNrs)
                    
    def draw_levels(self, fig=None, QuNrs_sep=['v'], level_length=0.8,
                    xlabel_pos='bottom',ylabel=True,yaxis_unit='MHz'):
        """This method draws all levels of the Electronic state sorted
        by certain Qunatum numbers.

        Parameters
        ----------
        fig : Matplotlib.figure object, optional
            Figure object into which the axes are drawn. The default is None which
            corresponds to a default figure.
        QuNrs_sep : list of str, optional
            Quantum numbers for separating all levels into subplots.
            By default the levels are grouped into subplots by the vibrational
            Quantum number, i.e. ['v'].
        level_length : float, optional
            The length of each level line. 1.0 corresponds to no space between
            neighboring level lines. The default is 0.8.
        xlabel_pos : str, optional
            Position of the xticks and their labels. Can be 'top' or 'bottom'.
            The default is 'bottom'.
        ylabel : bool, optional
            Wheter the ylabel should be drawn onto the y-axis.
        yaxis_unit : str or float, optional
            Unit of the y-axis. Can be either 'MHz','1/cm', or 'Gamma' for the
            natural linewidth. Alternatively, an arbitrary unit (in MHz) can be
            given as float. Default is 'MHz'.

        Returns
        -------
        coords : dict
            Dictionary with the coordinates of the single levels in the respective
            subplots. Two keys: 'axes' objects for every level index, and
            'xy' np.array of size 2 for the level coordinates within each subplot.
        """
        # check and verify the Quantum numbers for separation of the subplots QuNrs_sep
        QuNrs = self[0].QuNrs # the first states Quantum numbers (same as for all others)
        if len(QuNrs_sep) != 0:
            for QuNr_sep in QuNrs_sep:
                if not (QuNr_sep in QuNrs):
                    raise Exception('wrong input parameter')
        else:
            QuNrs_sep = [QuNrs[0]]
        
        # assign state indices to certain Quantum number tuples, i.e. certain sublpots
        QuNrs_sets = {}
        for l,st in enumerate(self.states):
            QuNr_set = tuple(st.__dict__[QuNr_sep] for QuNr_sep in QuNrs_sep) #what happens with loss state here?
            if QuNr_set in QuNrs_sets:
                QuNrs_sets[QuNr_set].append(l)
            else:
                QuNrs_sets[QuNr_set] = [l]

        # calculate frequency shifts of each state #these lines should be an extra method to be used in calc_freq() method!?!
        self._freq_arr = np.zeros(self.N)
        DF = self.get_freq()
        for l,st in enumerate(self.states):
            if st.is_lossstate: #leave this restriction here?!
                continue
            val = Levelsystem.state_in_DF(Levelsystem(),st,DF)
            if val != None: self._freq_arr[l] = val*1e6
        self._freq_arr *= 2*pi #make angular frequencies
        freq_arr = self._freq_arr/2/pi*1e-6 # in MHz  
            
        # frequency unit for y-axis
        if isinstance(yaxis_unit,str):
            if yaxis_unit == 'Gamma' and self.gs_exs == 'gs':
                warnings.warn("Gamma (natural decay rate) is not defined for an\
                             electronic ground state. So, 'MHz' is set instead.")
                ylabel_unit, yaxis_unit = 'MHz', 1.0
            elif yaxis_unit == 'Gamma' and self.gs_exs == 'exs':
                ylabel_unit, yaxis_unit = '$\Gamma$', self.Gamma
            else:
                ylabel_unit = yaxis_unit
                cm2MHz      = 299792458.0*100*1e-6
                yaxis_unit  = {'MHz':1.0, '1/cm':cm2MHz}[yaxis_unit]
        else:
            ylabel_unit = '{:.2f} MHz'.format(yaxis_unit)
        
        # create figure and subplot axes
        if fig == None:
            fig = plt.figure('Levels of {}'.format(self.label))
        gs_kw = dict(width_ratios=[len(inds) for inds in QuNrs_sets.values()])#,right=0.9,left=0.1,top=1.0)
        axs = fig.subplots(1, len(QuNrs_sets), gridspec_kw=gs_kw)
        if not isinstance(axs,Iterable): axs = [axs]
        
        # coordinates: axes objects for every level index, and xy level coords within each subplot
        coords = dict(axes=[None]*self.N,
                      xy=np.zeros((self.N,2)),
                      yaxis_unit=yaxis_unit)
        # draw levels and xticks
        if ylabel: axs[0].set_ylabel('Freq. [{}]'.format(ylabel_unit))
        for ax,QuNrs_set,inds in zip(axs,QuNrs_sets.keys(),QuNrs_sets.values()):
            if QuNrs_sep == [QuNrs[0]]:
                title = self.label
            else:
                title_bracket = ','.join(['{}={}'.format(QuNr_sep,QuNrs_set[i])
                                          for i,QuNr_sep in enumerate(QuNrs_sep)])
                title = '{}$({})$'.format(self.label,title_bracket)
            ax.set_xlabel(title)
            ax.xaxis.set_label_position(xlabel_pos)
            ax.xaxis.set_ticks_position(xlabel_pos)
            for i,ind in enumerate(inds):
                coords['xy'][ind,:] = i, freq_arr[ind]/yaxis_unit
                coords['axes'][ind] = ax
                ax.plot([i-level_length/2,i+level_length/2],
                        [freq_arr[ind]/yaxis_unit]*2,
                        color='k',linestyle='-',linewidth=1.)
            ax.set_xticks(np.arange(len(inds)))
            ax.set_xticklabels([str(ind) for ind in inds])
        
        return coords
        
    #%%
    def get_freq(self):
        """Function returns a list of matrices for nicely displaying
        the wavelengths between the vibrational transitions and the 
        frequencies between hyperfine transitions to conveniently specifying
        or modifying all participating transitions. These wavelengths and
        frequencies are loaded from the :py:class:`constants` module if
        available. Otherwise all wavelengths are set to 860e-9 and all other
        hyperfine frequencies to zero to be adjusted.

        Returns
        -------
        list of pandas.DataFrame and pandas.Series entries.
            list of matrices specifying the frequencies of the participating
            transitions.
        """
        if len(self._freq) != 0:
            return self._freq
        out = constants.get_DataFrame(self.const_dict,'HFfreq',gs_exs=self.label)
        if len(out) != 0:
            self._freq = out
        else:
            self._freq = self.DFzeros_without_mF()
        return self._freq
    
    def get_gfac(self):
        """Function returns a list of matrices for nicely displaying
        the g-factors of the ground states and the excited states respectively
        to conveniently specifying or modifying them. These g-factors are
        loaded from the :py:class:`constants` module if available. Otherwise
        g-factors are set to 0 to be adjusted.
        
        Returns
        -------
        list of pandas.Series entries.
            list of two matrices for the g-factors of the ground and excited states.
        """
        if len(self._gfac) != 0:
            return self._gfac
        out = constants.get_DataFrame(self.const_dict,'gfac',gs_exs=self.label)
        if len(out) != 0:
            self._gfac = out
        else:
            self._gfac = self.DFzeros_without_mF()
        return self._gfac
    
    #: hyperfine frequencies
    freq        = property(get_freq)
    #: g-factors
    gfac        = property(get_gfac)
    
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if self.N == 0: raise Exception('No states are defined/ included within Electronic State {}!'.format(self.label))
        if isinstance(index, (int, slice)): 
            return self.states[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.states[i] for i in index] 
    
    def __delitem__(self,index):
        """delete states using del system.levels[<normal indexing>], or delete all del system.levels[:]"""
        if self.properties_not_calculated:
            print('{} deleted!'.format(self.states[index]))
            del self.states[index]
        else:
            raise Exception('After any property is initialized, one can not delete states anymore')
    
    def __str__(self):
        """__str__ method is called when an object of a class is printed with print(obj)"""
        for i,st in enumerate(self.states):
            print('{:2d} {}'.format(i,st))
        Name = {'gs':'ground','exs':'excited','':''}[self.gs_exs]
        return "==> Electronic {} state {} with {:d} states in total".format(Name,self.label,self.N)
    
    def print_properties(self): 
        """Prints all relevant constants and properties of the composed levelsystem
        in a convenient way to modify them if needed afterwards.
        """
        n=40
        print('\n{s:{c}^{n}}'.format(s=self.label,n=n,c='-'))
        print('\ng-factors:',   self.gfac, sep='\n')
        print('\nfrequencies (in MHz):', self.freq, sep='\n')
    
    def DFzeros_without_mF(self): #is this important anymore? is required for get functions when no const_dict is available
        """Property returns two arrays containing the labels of all levels without
        the hyperfine magnetic sublevels quantum numbers mF.        
    
        Returns
        -------
        numpy.ndarray
            Quantum numbers of all ground levels without the magnetic sublevels mF.
        numpy.ndarray
            Quantum numbers of all excited levels without the magnetic sublevels mF.
        """
        self.properties_not_calculated = False
        QuNrs = self[0].QuNrs_without_mF
        rows = []
        for i,st in enumerate(self.states):
            if not st.is_lossstate:
                rows.append(tuple(st.__dict__[QuNr] for QuNr in QuNrs))
        Index= pd.MultiIndex.from_frame(pd.DataFrame(set(rows), columns=QuNrs))
        return pd.Series(0.0,index=Index)

    @property
    def N(self):
        '''Returns integer as number of defined :class:`State` instances.'''
        return len(self.states)

    @property
    def v_max(self):
        if len(self.states)==0:
            raise Exception('There are no levels defined! First, levels have to be added to',self.label)
        if not ('v' in self.states[0].QuNrs):
            raise Exception("There is no Quantum number 'v' defined for the states of {}".format(self.label))
        return max([st.v for st in self.states])
        
#%%
class ElectronicGrState(ElectronicState):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gs_exs = 'gs'
        
    def add_lossstate(self,v=None):
        """Adds a :class:`Lossstate` object to the self.entries
        list of this class only when no loss state is already included.

        Parameters
        ----------
        v : int, optional
            all ground state levels with the vibrational quantum number `v`
            and higher vibrational numbers are represented by a single loss state.
            Provided the default value None a loss state is added which is
            lying in the next higher vibrational manifold than the existing one
            in the already included ground levels.
        """
        if self.has_lossstate == False:
            if v == None:
                v = self.v_max+1
            QuNrs = { self.gs_exs : self.label, 'v' : v }
            self.states.append(State(is_lossstate=True,**QuNrs))
        else: print('loss state is already included')
    
    def del_lossstate(self):
        """Function simply deletes a loss state if one is existing in the defined
        level system."""
        if self.has_lossstate == True:
            index = None
            for i,st in enumerate(self.states):
                if st.is_lossstate:
                    index = i
                    break
            del self.states[index]
        else: print('There is no loss state included to be deleted')
        
    def print_remix_matrix(self): #old function! either move to Bfield or delete??
        """Print out the magnetic remixing matrix of the ground states by the
        usage of function :func:`System.magn_remix`.
        """
        from System import Bfield
        mat = Bfield().get_remix_matrix(self,0)
        for l1 in range(self.lNum):
            for l2 in range(self.lNum):
                if l2 == (self.lNum-1): end = '\n'
                else: 
                    end = ''
                if (l2+1) % 12 == 0: sep= '|'
                else: sep = ''     
                print(int(mat[l1,l2]),sep,end=end)
            if (l1 +1) % 12 == 0: print(self.lNum*2*'_')
        
    @property
    def has_lossstate(self):
        """Return True or False depending if a loss state is included in the ground levels."""
        for st in self.states:
            if st.is_lossstate: return True
        return False
#%%
class ElectronicExState(ElectronicState):
    def __init__(self,*args,Gamma=None,**kwargs):
        # Gamma is additional kwarg here!
        super().__init__(*args,**kwargs)
        self.gs_exs = 'exs'
        #: decay rate :math:`\Gamma` which is received from the function
        #: :func:`constants.Gamma`
        if Gamma:
            self.Gamma = Gamma
        else:
            Gamma = constants.get_DataFrame(self.const_dict,'Gamma',self.label)
            if len(Gamma) == 0:
                if self.verbose:
                    warnings.warn('Gamma has to be defined for the {} ElectronicState! For now it is set to 1 MHz by default!'.format(self.label))
                self.Gamma = 1.0 #1MHz
            else:
                self.Gamma = Gamma.iloc[0]
                
    def print_properties(self): 
        """Prints all relevant constants and properties of the composed levelsystem
        in a convenient way to modify them if needed afterwards.
        """
        super().print_properties()
        n=40
        print('\nGamma (in MHz):\n{} {}'.format(self.gs_exs, self.label), self.Gamma)
        
#%% #########################################################################
class State:
    def __init__(self,is_lossstate=False,**QuNrs):
        """Quantum state which contains all its quantum numbers. An electronic
        state instance :class:`ElectronicState` includes particularly such states
        of the same basis

        Parameters
        ----------
        is_lossstate : bool, optional
            determines whether the state has the function of a loss state which
            does not interact with any lasers but the populations can decay into
            this state. The branching into this state is given by the vibrational
            branching ratios, so it must contain the vibrational quantum number v.
            The default is False.
        **QuNrs : kwargs of int or float
            Quantum numbers of the state.
            
        Tip
        ---
        Like for the other classes, you can simply print the state instance::
            
            mystate = State(J=0.5,F=1,mF=0)
            print(mystate)
        """
        #: boolean variable determining if the state is a loss state
        self.is_lossstate = is_lossstate
        #: list of the all quantum numbers
        self.QuNrs = list(QuNrs.keys())
        for QuNr,value in QuNrs.items():
            if isinstance(value,float) and value.is_integer():
                QuNrs[QuNr] = int(value)
        self.__dict__.update(QuNrs)
    
    def copy(self):
        """Returns a deepcopy of this state's instance."""
        return deepcopy(self)
    
    def __eq__(self, other):
        if self.is_lossstate != other.is_lossstate:
            return False
        if len(self.QuNrs) != len(other.QuNrs):
            # return False
            two_sets = '\n--> state 1: {} <-> state 2: {}'.format(self.QuNrs,other.QuNrs)
            raise Exception('The two states have different sets of Quantum numbers!'+two_sets)
        else:
            for QuNr in self.QuNrs:
                if self.__dict__[QuNr] != other.__dict__[QuNr]:
                    return False
        return True
    
    def is_equal_without_mF(self, other):
        """Returns `True` if a state is equal to an other state neglecting
        different mF sublevels.

        Parameters
        ----------
        other : :class:`State`
            Other state to compare with.
        """
        if self.is_lossstate != other.is_lossstate:
            return False
        if len(self.QuNrs) != len(other.QuNrs):
            # return False
            raise Exception('The two states have different sets of Quantum numbers!')
        else:
            for QuNr in self.QuNrs:
                if QuNr == 'mF':
                    continue
                if self.__dict__[QuNr] != other.__dict__[QuNr]:
                    return False
        return True
    
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        if self.is_lossstate == True:
            str_lossstate = ' (loss state)'
        else:
            str_lossstate = ''
        return 'State : {}{}'.format(
            ', '.join(['{}={}'.format(QuNr,self.__dict__[QuNr]) for QuNr in self.QuNrs]),
            str_lossstate)
    @property
    def QuNrs_without_mF(self):
        '''Returns all the quantum numbers without mF'''
        return [QuNr for QuNr in self.QuNrs if QuNr != 'mF']
#%%
def get_constants_dict(name=''):
    def openjson(root_dir):
        with open(root_dir + name + ".json", "r") as read_file:
            data = json.load(read_file)
        return data
    if name:
        try:
            return openjson("./")
        except FileNotFoundError:
            script_dir = os.path.dirname(os.path.abspath(__file__)) #directory where this script is stored.
            # Using this directory path, the module System (and the others) can be imported
            # from an arbitrary directory provided that the respective path is in the PYTHONPATH variable.
            return openjson(script_dir + "\\")
    else:
        return {}