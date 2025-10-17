# -*- coding: utf-8 -*-
"""
This module contains all classes and methods to define all **states** and their
**properties** belonging to a certain Levelsystem.
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g
from scipy.constants import u as u_mass
from scipy.special import voigt_profile
from MoleCool.tools import dict2DF, get_constants_dict, auto_subplots
from collections.abc import Iterable
import matplotlib.pyplot as plt
import warnings
import os
import numbers
from copy import deepcopy
import pandas as pd
from sympy.physics.wigner import wigner_3j,wigner_6j
#%%
class Levelsystem:
    def __init__(self,load_constants=None,verbose=True):
        """Levelsystem consisting of :class:`ElectronicState` instances
        and methods to add them properly.
        These respective objects can be retrieved and also deleted by using the
        normal item indexing of a :class:`Levelsystem`'s object::
            
            from MoleCool import Levelsystem
            levels = Levelsystem()
            
            # define as ground state ('gs')
            levels.add_electronicstate('S12', 'gs')
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
        the function :meth:`print_properties()`. Afterwards the values in these
        pandas.DataFrames (here: vibrational branchings, transition wavelength,
        and g-factor) can be easily modified via `<DataFrame>.iloc[<index>]`.
        
        Tip
        ---
        Important properties within an instance :class:`Levelsystem` can be
        accessed with the ``get_<property>()`` methods or directly via the
        properties without ``get_`` in their names, like:
            
        * dMat          # electric dipole matrix
        * dMat_red      # reduced electric dipole matrix
        * vibrbranch    # vibrational branching ratios
        * wavelengths   # wavelengths (in nm) between the transitions
        """
        # unsorted list containing the labels of all added ElectronicStates:
        self.__ElSts_labels             = []
        self.verbose                    = verbose
        # loading the dictionary with the constants from json file
        self.load_constants             = load_constants
        self.const_dict                 = get_constants_dict(load_constants)
        # initialize all property instances
        self.reset_properties()
        
    def __getitem__(self,index):
        if isinstance(index, str):
            if index not in self.__ElSts_labels:
                raise ValueError(f"No ElectronicState '{index}' defined.")
            return self.__dict__[index]
        else:
            if self.N == 0:
                raise Exception('No states defined within the Levelsystem!')
            #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
            if isinstance(index, (int, slice)): 
                return self.states[index]
            #if indices are tuples instead (e.g. obj[1,3,2])
            return [self.states[i] for i in index]
    
    def __delitem__(self,label):
        """delete electronic states del system.levels[<electronic state label>], or delete all del system.levels[:]"""
        if type(label) != str:
            raise ValueError('index for an electronic state to be deleted must be provided as string of the label')
        if label not in self.__ElSts_labels:
            raise ValueError(f"No ElectronicState '{label}' defined.")
        self.__ElSts_labels.remove(label)
        del self.__dict__[label]
        self.reset_properties()
        print(f'ElectronicState {label} deleted and all properties resetted!')
        
    def add_all_levels(self,v_max):
        """Add all ground and excited states of a molecule conveniently with a
        loss state.

        Parameters
        ----------
        v_max : int
            all ground states with vibrational levels :math:`\\nu\\le` `v_max`
            and respectively all excited states up to `(v_max-1)` are added to
            the subclasses :class:`Groundstates` and :class:`Excitedstates`.
        """
        for key in list(self.const_dict['level-specific'].keys()):
            gs_exs,label = dict2DF.split_key(key)
            self.add_electronicstate(label, gs_exs)
            
            if gs_exs == 'gs':
                self[label].load_states(v=np.arange(0,v_max+.5,dtype=int))
                self[label].add_lossstate(v=v_max+1)
            elif gs_exs == 'exs':
                if v_max == 0: self[label].load_states(v=0)
                else:
                    self[label].load_states(v=np.arange(0,v_max-.5,dtype=int))
        
    def add_electronicstate(self,label,gs_exs,load_constants=None,**kwargs):
        """Add an electronic state (ground or excited state) as instance of
        the class :class:`ElectronicState` to this instance of
        :class:`Levelsystem`.

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
        
        Returns
        -------
        :class:`ElectronicState`
            Initialized electronic state instance.
        """
        if not label.isidentifier():
            raise ValueError('Please provide a valid variable/ instance name for `label`!')
        if (not load_constants) and self.load_constants:
            load_constants = self.load_constants
        if not ('verbose' in kwargs):
            kwargs['verbose'] = self.verbose
        if label in self.__ElSts_labels:
            raise Exception('There is already an ElectronicState {label} defined!')
        
        if gs_exs == 'gs':
            ElSt = ElectronicGrState(load_constants=load_constants,label=label,**kwargs)
        elif gs_exs == 'exs':
            ElSt = ElectronicExState(load_constants=load_constants,label=label,**kwargs)
        elif gs_exs == 'ims':
            ElSt = ElectronicImState(load_constants=load_constants,label=label,**kwargs)
        else:
            raise ValueError(("Please provide 'gs', 'exs', or 'ims' as `gs_exs` for "
                              "an electronic ground, excited or intermediate state"))
            
        self.__dict__[label] = ElSt
        self.__ElSts_labels.append(label)
        
        return ElSt
    #%% get functions
    # def load_save_DF(func):
    #     def wrapper(*args,**kwargs):
    #         print('warpper')
    #         inst = args[0]
    #         DFsaved, gs, exs = inst._load_qty(name='_dMat',*args,**kwargs)
    #         if len(DFsaved) != 0: return DFsaved
            
    #         DF = func(inst,gs,exs)
            
    #         inst._save_qty(DF,'_dMat',gs,exs)
    #         return DF
    #     return wrapper
    
    # @load_save_DF
    def get_dMat(self,gs=None,exs=None):
        """Return the electric dipole matrix. This matrix is
        either simply loaded from the .json file or constructed
        with the reduced electric dipole matrix given by the function
        :meth:`get_dMat_red()`.
        

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
        DFsaved, gs, exs = self._load_qty(name='_dMat',gs=gs,exs=exs)
        if len(DFsaved) != 0: return DFsaved
        
        #gs_exs_label = '-'.join([gs,exs]) #df1.index.to_frame()['gs'].iloc[0]#ß? #df1.index.names.index('gs') #df1.columns.to_frame()['exs'].iloc[0]
        DF_dMat         = dict2DF.get_DataFrame(self.const_dict,'dMat',gs=gs,exs=exs)
        DF_dMat_red     = dict2DF.get_DataFrame(self.const_dict,'dMat_red',gs=gs,exs=exs)
        DF_branratios  = dict2DF.get_DataFrame(self.const_dict,'branratios',gs=gs,exs=exs)
        if len(DF_dMat) != 0:
            dMat = DF_dMat
        elif (len(DF_dMat_red) == 0) and (len(DF_branratios) !=0):
            self._branratios = DF_branratios
            if self.verbose: warnings.warn('No dipole matrix or reduced dipole matrix found, '
                                           + 'so the dipole matrix is constructed from the given '
                                           + 'branching ratios only with positive values!')
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
                    for index2,row2 in row1.items():
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
            dMat = dMat.fillna(0.0)
        
        self._save_qty(dMat,name='_dMat',gs=gs,exs=exs)
        return dMat
    
    def get_dMat_red(self,gs=None,exs=None):
        """Return the reduced electric dipole matrix. This matrix is
        either simply loaded from the .json file or constructed
        with the electric dipole matrix given by the function :meth:`dMat_red`.
        If no dipole matrix or a reduced one is available a new reduced matrix
        is constructed with only ones for each transition which can be adjusted
        afterwards.
        
        Returns
        -------
        pandas.DataFrame
            Reduced electric dipole matrix.
        """
        DFsaved, gs, exs = self._load_qty(name='_dMat_red',gs=gs,exs=exs)
        if len(DFsaved) != 0: return DFsaved
                
        DF_dMat_red = dict2DF.get_DataFrame(self.const_dict,'dMat_red',gs=gs,exs=exs)
        if len(DF_dMat_red) != 0:
            dMat_red = DF_dMat_red
            # must be updated below!
        elif len(dict2DF.get_DataFrame(self.const_dict,'dMat',gs=gs,exs=exs)) != 0:
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
            dMat_red = pd.DataFrame(1.0,index=self[gs].DFzeros_without_mF().index,
                                    columns=self[exs].DFzeros_without_mF().index)
            if self.verbose:
                warn_txt = 'There is no dipole matrix or reduced dipole matrix available!' + \
                    'So a reduced matrix has been created only with ones:\n{}'.format(dMat_red)
                warnings.warn(warn_txt)
        
        self._save_qty(dMat_red,name='_dMat_red',gs=gs,exs=exs) 
        return dMat_red

    def get_vibrbranch(self,gs=None,exs=None):
        """Return a matrix for the vibrational branching ratios between
        vibrational excited levels with :math:`\\nu` and ground levels wth
        :math:`\\nu'`.This matrix is either simply loaded from the .json file
        or constructed with the same branching ratios
        for all transitions.        

        Returns
        -------
        pandas.DataFrame
            vibrational branching ratios matrix.
        """
        DFsaved, gs, exs = self._load_qty(name='_vibrbranch',gs=gs,exs=exs)
        if len(DFsaved) != 0: return DFsaved
        
        DF = dict2DF.get_DataFrame(self.const_dict,'vibrbranch',gs=gs,exs=exs)
        if len(DF) == 0:
            DF_gs = self[gs].DFzeros_without_mF()
            DF_exs = self[exs].DFzeros_without_mF()
            DF = pd.DataFrame(1.0,
                              index=DF_gs.index.droplevel([QuNr for QuNr in DF_gs.index.names
                                                           if QuNr not in ['gs','exs','ims','v']]).drop_duplicates(),
                              columns=DF_exs.index.droplevel([QuNr for QuNr in DF_exs.index.names
                                                              if QuNr not in ['gs','exs','ims','v']]).drop_duplicates())
            DF /= DF.sum(axis=0)
            
        self._save_qty(DF,name='_vibrbranch',gs=gs,exs=exs) 
        return DF
    
    def get_transdipmoms(self):
        """Return transition dipole moments between electronic states.

        Returns
        -------
        pandas.DataFrame
            Transition dipole moments.
        """
        if len(self._transdipmoms) == 0:
            DF = dict2DF.get_DataFrame(self.const_dict,'transdipmoments')
            if len(DF) == 0:
                DF = pd.DataFrame(1.0,index=self.grstates_labels,
                                  columns=self.exstates_labels)
        
            self._transdipmoms = DF
            
        ImSt_inds = [(i,j) for j,col in enumerate(self._transdipmoms.columns)
                     for i,ind in enumerate(self._transdipmoms.index) if col==ind]
        for ImSt_ind in ImSt_inds:
            self._transdipmoms.iloc[ImSt_ind] = 0.0
        
        return self._transdipmoms
    
    def get_wavelengths(self,gs=None,exs=None):
        """Return a list of matrices for nicely displaying
        the wavelengths between the vibrational transitions and the 
        frequencies between hyperfine transitions to conveniently specifying
        or modifying all participating transitions. These wavelengths and
        frequencies are loaded from the .json file if
        available. Otherwise all wavelengths are set to 860e-9 and all other
        hyperfine frequencies to zero to be adjusted.

        Returns
        -------
        list of pandas.DataFrame and pandas.Series entries.
            list of matrices specifying the frequencies of the participating
            transitions.
        """
        DFsaved, gs, exs = self._load_qty(name='_wavelengths',gs=gs,exs=exs)
        if len(DFsaved) != 0: return DFsaved
                
        DF = dict2DF.get_DataFrame(self.const_dict,'vibrfreq',gs=gs,exs=exs)
        if len(DF) == 0:
            DF = pd.DataFrame(860.0,index=self[gs].DFzeros_without_mF().index,
                              columns=self[exs].DFzeros_without_mF().index)
        
        self._save_qty(DF,name='_wavelengths',gs=gs,exs=exs) 
        return DF
    
    def _load_qty(self,name,gs=None,exs=None):
        if gs == None:  gs =    self.grstates_labels[0]
        if exs == None: exs =   self.exstates_labels[0]
        
        dic = self.__dict__[name]
        if gs in dic:
            if exs in dic[gs]:
                if len(dic[gs][exs]) != 0:
                    return dic[gs][exs], gs, exs
        return {}, gs, exs
    
    def _save_qty(self,value,name,gs,exs):
        dic = self.__dict__[name]
        if gs in dic:
            dic[gs][exs] = value
        else:
            dic[gs] = {exs: value}    
    
    #: electric dipole matrix
    dMat        = property(get_dMat)
    #: reduced electric dipole matrix
    dMat_red    = property(get_dMat_red)
    #: vibrational branching ratios
    vibrbranch  = property(get_vibrbranch)
    #: transition wavelengths and frequencies
    wavelengths = property(get_wavelengths)
    #: transition dipole moments
    transdipmoms= property(get_transdipmoms)
    #%% calc functions for the rate and optical Bloch equations
    def calc_dMat(self):
        """
        Calculate matrix elements of the electric dipole moment operator.
        In contrast to the other functions :meth:`get_dMat()` or
        :meth:`get_dMat_red()`, this method calculates the normalized electric
        dipole matrix as numpy.ndarray ready to be directly called and used
        for the methods :meth:`~.System.System.calc_rateeqs` and
        :meth:`~.System.System.calc_OBEs`.
        This matrix includes also the vibrational branching ratios and handles
        the loss state in a correct way and is not meant to be modified.

        Returns
        -------
        numpy.ndarray
            fully normalized electric dipole matrix.
        """
        
        if np.all(self.__dMat_arr) != None: return self.__dMat_arr
        self.__dMat_arr = np.zeros((self.lNum,self.uNum,3))
        
        DF_tdm = self.get_transdipmoms()
        DF_tdm /= np.sqrt((DF_tdm**2).sum(axis=0))
        #levels._dMat.xs((1.5,2,-1),level=('J','F','mF'),axis=0,drop_level=True).xs((0.5,1,-1),level=("J'","F'","mF'"),axis=1,drop_level=True)
        N_grstates = 0
        for Grs_lab, Grs in zip(self.grstates_labels, self.grstates):
            N_exstates = 0
            for Exs_lab,Exs in zip(self.exstates_labels, self.exstates):
                if Grs == Exs: continue
                DF_vb   = self.get_vibrbranch(gs=Grs_lab, exs=Exs_lab)
                val_tdm = DF_tdm.loc[Grs_lab,Exs_lab]
                if 'v' in Grs[0].QuNrs:
                    DF_vb   = DF_vb.iloc[np.argwhere(DF_vb.index.get_level_values('v') < Grs.v_max+0.1)[:,0]]
                DF_vb  /= DF_vb.sum(axis=0) # normalized DF_vb must be multiplied afterwards with a factor due to transition dipole moment
                
                DF_dMat = self.get_dMat(gs=Grs_lab, exs=Exs_lab)
                DF_dMat /= np.sqrt((DF_dMat**2).sum(axis=0))
                DF_dMat = DF_dMat.fillna(0.0)
                for l,gr in enumerate(Grs.states):
                    for u,ex in enumerate(Exs.states):
                        val_vb = self.val_states_in_DF(gr,ex,DF_vb)
                        if gr.is_lossstate:
                            #the q=+-1,0 entries squared of the dMat are summed in the last line of the equations set in the Fokker-Planck paper.
                            # So for the loss state which should not interact with other levels, it doesn't matter which q component of the sum in the last line is contributing.
                            self.__dMat_arr[N_grstates+l, N_exstates + u, :] = np.array([1,0,0])*np.sqrt(val_vb)
                        else:
                            val_dMat = self.val_states_in_DF(gr,ex,DF_dMat)
                            pol = ex.mF-gr.mF
                            if (abs(pol) <= 1) and (val_vb != None):
                                self.__dMat_arr[N_grstates+l, N_exstates+u, int(pol)+1] = val_tdm*val_dMat*np.sqrt(val_vb)
                N_exstates += Exs.N
            N_grstates += Grs.N
        
        self.__dMat_arr /= np.sqrt((self.__dMat_arr**2).sum(axis=(2,0)))[None,:,None]
        # np.nan_to_num(self.__dMat_arr,copy=False)
        return self.__dMat_arr
    
    def calc_branratios(self):
        """Calculate fully normalized branching ratios using the dipole
        matrix calculated in the function :meth:`calc_dMat()`
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
        """Calculate the angular absolute frequency **differences**
        between **all** levels included in this class using the wavelengths and
        frequencies specified by the function :meth:`~ElectronicState.get_freq()`.
        These values are returned as numpy array ready to be directly called
        and used for the functions :meth:`~.System.System.calc_rateeqs()`
        and :meth:`~.System.System.calc_OBEs()`.
        
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
                        val = self.val_states_in_DF(gr,ex,DF)
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
                    val = self.val_state_in_DF(st,DF)
                    if val != None:
                        if i0 == 0:   self.__freq_arr[N_ElSts+i_st,:] -= val*1e6
                        elif i0 == 1: self.__freq_arr[:,N_ElSts+i_st] += val*1e6
                N_ElSts += ElSt.N
            
        self.__freq_arr *= 2*pi #make angular frequencies
        return self.__freq_arr
    
    def val_states_in_DF(self,st1,st2,DF): #move this and the other function outside of the class with an kwarg verbose.
        ind_names = list(DF.index.names)
        col_names = list(DF.keys().names)
        for index1,row1 in DF.iterrows():
            if len(ind_names) == 1: index1 = (index1,) #index1 must be a iterable tuple even if it containts only 1 element
            for index2,row2 in row1.items():
                if len(col_names) == 1: index2 = (index2,)
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
            return self.val_states_in_DF(st1_,st2_,DF)
        elif self.verbose:
            warnings.warn('No value in DF found for {}, {}'.format(st1,st2))
        return None
    
    def val_state_in_DF(self,st,DF):
        ind_names = list(DF.index.names)
        for index1,row1 in DF.items():
            allTrue = [True]
            for i,ind_name in enumerate(ind_names):
                if not np.all(allTrue): break
                allTrue.append(index1[i] == st.__dict__.get(ind_name,None))
            if np.all(allTrue):
                return row1
            
        if ('v' in st.QuNrs) and (st.v > 0) and (not st.is_lossstate):
            st_     = st.copy()
            st_.v   = 0
            return self.val_state_in_DF(st_,DF)
        elif self.verbose:
            warnings.warn('No value in DF found for {}'.format(st))
        return None
        
    
    def calc_muMat(self):
        """Calculate the magnetic dipole moment operator matrix for **all** levels
        included in this class using the g-factors specified by the function
        :meth:`~ElectronicState.get_gfac()`.
        These values are returned as numpy array ready to be directly called
        and used for the function :meth:`~.System.System.calc_OBEs()`.
        
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
                if (i0 == 1) and (ElSt.gs_exs == 'ims'): continue
                DF = ElSt.get_gfac()
                for i1, st1 in enumerate(ElSt.states):
                    if st1.is_lossstate:
                        continue # all elements self._muMat[i0][i1,N:ElSt.N,:] remain zero
                    val = self.val_state_in_DF(st1,DF)
                    F,m = st1.F, st1.mF
                    for i2, st2 in enumerate(ElSt.states):
                        if st2.is_lossstate:
                            continue # all elements self._muMat[i0][i1,i2,:] remain zero
                        if not st1.is_equal_without_mF(st2):
                            continue
                        n = st2.mF
                        for q in [-1,0,1]:
                            if val != None:
                                self._muMat[i0][N+i1,N+i2,q+1] = -val* (-1)**(F-m)* \
                                    np.sqrt(F*(F+1)*(2*F+1)) * float(wigner_3j(F,1,F,-m,q,n))
                N += ElSt.N
        return self._muMat
    
    def calc_M_indices(self):
        """Calculate the indices determining all hyperfine states within
        a specific F or F' of the ground or excited state. These values are
        used in the function :meth:`~.System.System.calc_OBEs()` when looping
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
        """Calculate the natural decay rate Gamma (in angular frequency)
        for each single excited state.

        Returns
        -------
        numpy.ndarray
            Gamma array of length uNum as angular frequency [Hz].
        """
        if np.all(self._Gamma) != None:
            return self._Gamma
        else:
            self._Gamma = np.array([ElSt.Gamma for ElSt in self.exstates
                                    for i in range(ElSt.N)]) * 2*pi * 1e6
            return self._Gamma
        
    def calc_all(self):
        """Calculate all level properties once in the beginning, so that they
        are stored and calling the method
        :meth:`~MoleCool.System.System.calc_OBEs()` multiple times doesn't
        require additional computation time.
        """
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
    
    def transitions_energies_strengths(self, include_forbidden=False,
                                       gs=None, exs=None, use_calc_props=True):
        """Yield the transition strengths and energies without including all mF
        levels. This is e.g. used by the method :meth:`plot_transition_spectrum`.
        The strengths are solely calculated using the property :meth:`dMat_red`.
        
        Parameters
        ----------
        include_forbidden : bool, optional
            Whether to include also the transitions with vanishing branchings.
            The default is False.
        gs : str, optional
            Electronic ground state label. The default is None.
        exs : str, optional
            Electronic excited state label. The default is None.
        use_calc_props : bool, optional
            If True, the properties like the dipole matrix and all transition
            energies are calculated and then used to plot the spectrum with
            exactly the values that are used for the simulation. If False,
            the reduced dipole matrix property is directly used and the
            wavelengths property is not taken into account.

        Returns
        -------
        Es : np.ndarray
            Energies in MHz.
        branchings : np.ndarray
            reduced branching ratios.
        states : list(tuple)
            list of tuple pairs containing the ground and excited state for each
            transition energy and branching.
        """
        if gs == None:  gs =    self.grstates_labels[0]
        if exs == None: exs =   self.exstates_labels[0]
        
        states_sets = dict()
        inds = dict()
        for j,ElSt_label in enumerate([gs,exs]):
            states = []
            indices = []
            for i,st in enumerate(self[ElSt_label].states):
                if not np.any([st.is_equal_without_mF(st2) for st2 in states]):
                    states.append(st)
                    indices.append(i)
            states_sets[ElSt_label] = states
            inds[ElSt_label] = indices
        
        if use_calc_props:
            inds2 = []
            for j, ElSt in enumerate([self[gs], self[exs]]):
                inds2.append([[i for i,st in enumerate(ElSt) if st0.is_equal_without_mF(st)]
                              for st0 in states_sets[ElSt.label]])
        
        Es = []
        branchings = []
        states = []
        ElGr, ElEx = list(states_sets.keys())
        for i,gr in enumerate(states_sets[ElGr]):
            if gr.is_lossstate:
                continue
            for j,ex in enumerate(states_sets[ElEx]):
                if use_calc_props:
                    branching = self.calc_branratios()[np.ix_(inds2[0][i],inds2[1][j])].sum()
                    E = ((self.calc_freq()/2/pi)*1e-6)[inds[gs][i],inds[exs][j]]
                else:
                    branching = self.val_states_in_DF(gr, ex, self.get_dMat_red(gs=gs,exs=exs)**2)
                    E = self.val_state_in_DF(ex, self[ElEx].get_freq()) \
                        - self.val_state_in_DF(gr, self[ElGr].get_freq())
                if include_forbidden or branching:
                    branchings.append(branching)
                    Es.append(E)
                    states.append((gr,ex))
                    
        return Es, branchings, states

    def plot_transition_spectrum(self, std=0, xaxis=[], xaxis_ext=5, N_points=500,
                                 wavelengths=[], relative_to_wavelengths=False,
                                 kwargs_single=dict(), kwargs_sum=dict(ls='-', color='k'),
                                 plot_single=True, plot_sum=True,
                                 ax=None, legend=True, QuNrs=[['F'],['F']],
                                 exs=[], subplot_sep=200., E_unit='MHz',
                                 E_offset=0, kwargs_trans_ener_stre=dict()):
        """Plot the transition spectrum with Voigt profiles using the energies
        and strengths from :meth:`transitions_energies_strengths()`.

        Parameters
        ----------
        std : float, optional
            Standard deviation resulting in the Gaussian broadening of the voigt
            profile in MHz. The Lorentzian broadening is always given by the
            natural linewidth of the electronic excited state. The default is 0.
        xaxis : list or np.ndarray, optional
            xaxis data points. The default is [].
        xaxis_ext : float, optional
            if not xaxis is given, the range of the xaxis, given by the
            lowest and highest transition frequency, is extended by this
            factor multiplied by the sum of the Gaussian and Lorentzian broadening.
            The default is 5.
        N_points : int, optional
            number of points plotted. The default is 500.
        wavelengths : list, optional
            wavelengths that should be plotted within the range ``subplot_sep``.
            By default all available transition wavelengths are used.
        relative_to_wavelengths : bool, optional
            Whether the x-axis should be plotted in absolute frequency units or
            relative to ``wavelengths``.
        kwargs_single : dict, optional
            keyword arguments used in ``plt.plot()`` for the single transition lines.
            The default is dict().
        kwargs_sum : dict, optional
            same as ``kwargs_single`` but for the sum.
            The default is dict(ls='-', color='k').
        plot_single : bool, optional
            Whether to plot single transition lines. The default is True.
        plot_sum : bool, optional
            Whether to plot the sum of all transition lines. The default is True.
        ax : ``matplotlib.pyplot.axis`` or list of ``axis``, optional
            axis or multiple axes to put the plot(s) on. The default is None.
        legend : bool, optional
            Whether to show legend. The default is True.
        QuNrs : list(list), optional
            QuNrs of ground and excited state used for labelling and legend.
            The default is [['J','F'],['F']].
        exs : list(str), optional
            List of excited states to be plotted (see :meth:`transitions_energies_strengths`).
            By default only the first excited electronic state is used.
        subplot_sep : float, optional
            Defines the range of the plotted x-axis and the separation for the 
            automatic inclusion of all wavlengths (see parameter ``wavelengths``)
            in units of :meth:`ElectronicExState.Gamma`. Default is 200.
        E_unit : str, optional
            Unit of the x-axis to be plotted.
            Can be one of ``['GHz','MHz','kHz','Hz','Gamma']``. Default is 'MHz'.
        E_offset : float, optional
            Energy offset on the xaxis. The default is 0.
        kwargs_trans_ener_stre : dict, optional
            keyword arguments for :meth:`transitions_energies_strengths`, e.g.
            gs as label used for the electronic ground state.

        Returns
        -------
        ax : matplotlib.pyplot.axes or list for multiple plots
            axis of the plot.
        data : dict(np.ndarray) or list for multiple plots
            raw data thats plotted. Includes the single transitions, the sum and
            the x axis data.
        """
        if not exs:
            exs = [self.exstates_labels[0]]
        
        if len(exs)>1:
            raise Exception('Multiple excited electronic states are not supported yet.')
        else:
            ExSt = exs[0]
            kwargs_trans_ener_stre['exs'] = ExSt
        
                
        if not 'exs' in kwargs_trans_ener_stre:
            kwargs_trans_ener_stre['exs'] = self.exstates_labels[0]
        exs     = kwargs_trans_ener_stre['exs']
            
        out     = self.transitions_energies_strengths(**kwargs_trans_ener_stre)
        for i,el in enumerate(out[0]):
            out[0][i] *= 1e6 # from MHz to Hz
        
        
        Gamma   = self[ExSt].Gamma*1e6 # natural linewidth in Hz
        subplot_sep *= Gamma
        x_ext   = (Gamma+std)*xaxis_ext # extension on the x axis
        fac_x   = {'GHz':1e9, 'MHz':1e6, 'kHz':1e3, 'Hz':1, 'Gamma':Gamma}[E_unit]
        
        # get wavelengths of all transition blocks that span subplot_sep*Gamma
        if not list(wavelengths):
            fsort   = np.sort(out[0]) # sorted
            
            inds    = [i+1 for i,df in enumerate(np.diff(fsort)) if df > subplot_sep]
            inds    = [0, *inds]
        
            fmean   = np.array([ fsort[i:j].mean() for i,j in zip(inds, inds[1:]+[None])])
            
            wavelengths = c/fmean
        
        
        # generate subplots
        axs     = auto_subplots(len(wavelengths), axs=[] if ax==None else ax)
        data_list = []
        
        for j,(ax,wavelength) in enumerate(zip(axs,wavelengths)):
            
            xconv   = lambda x: (x - c/wavelength*int(bool(relative_to_wavelengths)))/fac_x
            
            inds    = np.argwhere((out[0]>c/wavelength-subplot_sep) \
                                  & (out[0]<c/wavelength+subplot_sep))[:,0]
            
            out_cut = [[out_i[i] for i in inds] for out_i in out]
            if not out_cut[0]:
                continue
            if len(xaxis) == 0 or j>0:
                xaxis   = np.linspace(min(out_cut[0])-x_ext, max(out_cut[0])+x_ext, N_points)
                
            spectrum = np.zeros(len(xaxis))
            y_arr    = []
            
            
            norm = voigt_profile(np.linspace(-(std+Gamma),+(std+Gamma),500), std, Gamma/2).max()
            for E,bran,(gr,ex) in zip(*out_cut):
                # Gaussian and Lorentzian (note factor of 2) broadening in Voigt profile
                voigt   = voigt_profile(E-xaxis, std, Gamma/2)
                y       = bran*voigt/norm
                y_arr.append(y)
                label   = None
                if legend:
                    strings = [[str(st.__dict__[QuNr]) for QuNr in QuNrs_] for st,QuNrs_ in zip([gr,ex],QuNrs)]
                    label = ', '.join(strings[0]) + '$\\rightarrow$' + ', '.join(strings[1])
                                                                       
                spectrum += y
                if plot_single:
                    ax.plot(xconv(xaxis)+E_offset, y, label=label, **kwargs_single)
                    
            if plot_sum:
                ax.plot(xconv(xaxis)+E_offset, spectrum, **kwargs_sum)
                
            xlabel = f'Frequency ({E_unit})'
            if relative_to_wavelengths:
                xlabel += f" at {wavelength*1e9:f} nm"
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Line strength')
            if legend:
                ax.legend(title=', '.join(QuNrs[0]) + '$\\rightarrow$' + "', ".join(QuNrs[1]) + "'")
            data_list.append(dict(x=xaxis+E_offset, sum=spectrum, single=np.array(y_arr)))
        
        if len(wavelengths) == 1:
            return axs[0], data_list[0]
        else:
            return axs, data_list

    def print_properties(self): 
        """Print all relevant constants and properties of the composed levelsystem
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
        print('\ntransition dipole moments:', self.transdipmoms, sep='\n')
        for Grs_lab, Grs in zip(self.grstates_labels, self.grstates):
            for Exs_lab,Exs in zip(self.exstates_labels, self.exstates):
                if Grs == Exs: continue # skip for intermediate states
                print('\n{s:{c}^{n}}'.format(s=Grs_lab + ' <- ' + Exs_lab,n=n,c='-'))
                print('\ndipole matrix:',                 self.get_dMat(gs=Grs_lab,exs=Exs_lab),
                      '\nvibrational branching:',         self.get_vibrbranch(gs=Grs_lab,exs=Exs_lab),
                      '\nwavelengths (in nm):',           self.get_wavelengths(gs=Grs_lab,exs=Exs_lab),
                      sep='\n')
        
    def check_config(self,raise_Error=False):
        """Check the configuration of the Levelsystem to be used in calculating
        laser cooling dynamics. E.g. involves to check whether the states are 
        correctly defined.
        
        Parameters
        ----------
        raise_Error : bool, optional
            If the configuration is not perfect, this method raises an error message
            or only prints a warning depending on `raise_Error`. The default is False.
        """
        Err_str = 'Electronic state {} contains no levels!'
        for ElSt in self.electronic_states:
            if ElSt.N == 0:
                if raise_Error: raise Exception(Err_str.format(ElSt.label))
                else: warnings.warn(Err_str.format(ElSt.label))
                
    def reset_properties(self):
        """Reset and initialize all property objects (e.g. dMat, dMat_red,
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
        self._transdipmoms              = []
        # The following variables with two underscores are only for internal use inside the class
        self.__dMat_arr                 = None
        self.__branratios_arr           = None
        self.__freq_arr                 = None
        self._muMat                     = None
        self._M_indices                 = None
        self._Gamma                     = None

    @property
    def description(self):
        """str: Display a short description with the number of included state objects."""
        return "{:d}+{:d} - Levelsystem".format(self.lNum,self.uNum)
    
    @property
    def states(self):
        """Return a combined list of all state objects defined in the individual
        electronic states.
        """
        return [st for ElSt in self.electronic_states for st in ElSt.states]
    
    @property
    def lNum(self):
        '''Return the total number of states defined in the ground electronic
        states as an integer.'''
        return sum([ElSt.N for ElSt in self.grstates])
    
    @property
    def uNum(self):
        '''Return the total number of states defined in the excited electronic
        states as an integer.'''
        return sum([ElSt.N for ElSt in self.exstates])
    
    @property
    def iNum(self):
        '''Return the total number of states defined in all intermediate electronic
        states as an integer.'''
        return sum([ElSt.N for ElSt in self.grstates if ElSt.gs_exs == 'ims']) 
    
    @property
    def N(self):
        '''Return the total number of unique states defined in all electronic
        states as an integer, i.e. N = :meth:`lNum` + :meth:`uNum` - :meth:`iNum`.'''
        return self.lNum + self.uNum -self.iNum
    
    @property
    def grstates(self):
        '''Return a list containing all defined instances of ground electronic
        states (:class:`ElectronicGrState`).'''
        ElSts = [self[label] for label in self.__ElSts_labels] # unsorted
        return [*[ElSt for ElSt in ElSts if ElSt.gs_exs == 'gs'],
                *[ElSt for ElSt in ElSts if ElSt.gs_exs == 'ims']]
    
    @property
    def exstates(self):
        '''Return a list containing all defined instances of excited electronic
        states (:class:`ElectronicExState`).'''
        ElSts = [self[label] for label in self.__ElSts_labels] # unsorted
        return [*[ElSt for ElSt in ElSts if ElSt.gs_exs == 'exs'],
                *[ElSt for ElSt in ElSts if ElSt.gs_exs == 'ims']]
    
    @property
    def electronic_states(self):
        '''Return a list containing all defined instances of electronic
        states, i.e. stacking list of :meth:`grstates` and :meth:`exstates`.'''
        return [*self.grstates,*[ExSt for ExSt in self.exstates if ExSt not in self.grstates]] 
    
    @property
    def grstates_labels(self):
        '''Return a list containing the labels of all ground electronic states'''
        return [ElSt.label for ElSt in self.grstates]
    
    @property
    def exstates_labels(self):
        '''Return a list containing the labels of all excited electronic states'''
        return [ElSt.label for ElSt in self.exstates]
   
    @property
    def Isat(self):
        """Calculate the two-level saturation intensity in W/m^2 for all
        transitions.
        """
        return pi*c*h*self.calc_Gamma()[None,:]/(3*( 2*pi*c/self.calc_freq() )**3)
    
    def Isat_eff(self,GrSt,ExSt):
        """Calculate the effective multi-level saturation intensity in W/m^2
        between two electronic states whose states are all coupled together.
        This quantity can be derived by rearranging the rate equations into a general
        approximate expression for the scattering rate. Here, the assumptions
        that all detunings are equal, all intensities are equal, and all excited
        states are equally populated are used.
        """
        lNum = self[GrSt].N
        uNum = self[ExSt].N
        i0,i1 = self.index_ElSt(GrSt,'gs'), self.index_ElSt(ExSt,'exs')
        return 2*lNum**2/(lNum+uNum)*self.Isat[i0:i0+lNum, i1:i1+uNum]
   
    def index_ElSt(self,ElSt,gs_exs=None,include_Ngrs_for_exs=False):
        """Return the total index of the first state belonging to a certain
        electronic state. For example for two electronic excited states with each
        3 single states defined, the method gives 0 and 3 for the first and second
        electronic excited state, respectively.

        Parameters
        ----------
        ElSt : str or :class:`ElectronicState`
            Electronic state.
        gs_exs : str, optional
            specify whether ElSt is a ground or excited state.
            The default is None so that its type is automatically inferred.
        include_Ngrs_for_exs : bool, optional
            specifies whether the total number of ground state levels is added
            for an excited state. Default is False.

        Returns
        -------
        int
            total index of first level state defined in the ElectronicState `ElSt`
        """
        if isinstance(ElSt, str):
            ElSt = self[ElSt]
        if gs_exs == None:
            gs_exs = {True: 'gs', False: 'exs'}[ElSt in self.grstates]
        
        ElSts = {'gs':self.grstates, 'exs':self.exstates}[gs_exs]  
        index = sum([ElSts[i].N for i in range(ElSts.index(ElSt))])
        if include_Ngrs_for_exs and gs_exs=='exs':#ElSt in self.exstates:
            index += self.lNum
        return index
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
        self.N0    = [] # initial population

    def add(self,**QuNrs):
        """Add an instance of :class:`State` to this electronic state.
        
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
            if np.any(np.abs(mF) > F):
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
        """Check if a state exists in this electronic state."""
        #test if a given state already exists in the ElectronicState and returns boolean
        for st in self.states:
            if st == state:
                return True
        return False
    
    def load_states(self,**QuNrs):
        """Load all states from the corresponding ``.json`` file that match the
        given ``QuNrs`` as keyword arguments.
        """
        if len(self.const_dict) == 0:
            raise Exception('There is no constants dictionary available to load states from!')
        if isinstance(QuNrs.get('v',None),Iterable):
            for v_i in QuNrs['v']:
                QuNrs['v'] = v_i
                self.load_states(**QuNrs) #recursively calling this method with no 'v' Iterable
        else:
            list_of_dicts = dict2DF.get_levels(dic=self.const_dict,gs_exs=self.label,**QuNrs)
            if not list_of_dicts:
                text = 'No pre-defined states found for electronic state {} with: {}'.format(
                    self.label, ', '.join(['{}={}'.format(key,QuNrs[key]) for key in QuNrs]))
                if 'v' in QuNrs:
                    # set vibrational quantum number to 0 and try to import constants
                    v_notfound = QuNrs['v']
                    QuNrs['v'] = 0
                    list2_of_dicts = dict2DF.get_levels(dic=self.const_dict,gs_exs=self.label,**QuNrs)
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
                    
    def draw_levels(self, fig=None, QuNrs_sep=[], level_length=0.8,
                    xlabel_pos='bottom',ylabel=True,yaxis_unit='MHz'):
        """Draw all levels of the Electronic state sorted by certain
        Quantum numbers.

        Parameters
        ----------
        fig : Matplotlib.figure object, optional
            Figure object into which the axes are drawn. The default is None which
            corresponds to a default figure.
        QuNrs_sep : list of str, optional
            Quantum numbers for separating all levels into subplots.
            For example the levels can be grouped into subplots by the vibrational
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
            val = Levelsystem.val_state_in_DF(Levelsystem(),st,DF)
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
    
    def set_init_pops(self, QuNrpops):
        """Set initial population of the levels as a starting point for the
        simulations.

        Parameters
        ----------
        QuNrpops : dict
            Specifies the population of the levels with certain Quantum numbers.
            E.g. `QuNrpos={'v=0':0.8, 'v=1,J=0.5':0.1}` implies that all levels
            that have the Quantum number `v=0` share a total population of 0.80
            and the levels with `v=1` and `J=0.5` share 0.10 whereas the
            populations of the remaining levels are set to zero.
            So, the sum of all populations specified can't be larger than 1.0.
        """
        N0 = np.zeros(self.N)
        if sum(QuNrpops.values()) > 1:
            raise ValueError('Sum of QuNrpops values can not be larger than 1.')
        
        inds_used = [] # level indices that are already used for population distribution
        for QuNrvals, pop in QuNrpops.items():
            QuNrvals_ = dict( ( pair.split('=')[0], float(pair.split('=')[1]) )
                            for pair in QuNrvals.split(','))
            inds = [i for i,st in enumerate(self)
                    if st.check_QuNrvals(**QuNrvals_) and i not in inds_used]
            if len(inds) != 0:
                N0[inds] = pop/len(inds)
                inds_used += inds
            else:
                warnings.warn(f'No states found for QuNrs {QuNrvals}')
        self.N0 = N0
        
    #%%
    def get_freq(self):
        """Return a list of matrices for nicely displaying
        the wavelengths between the vibrational transitions and the 
        frequencies between hyperfine transitions to conveniently specifying
        or modifying all participating transitions. These wavelengths and
        frequencies are loaded from the .json file if available.
        Otherwise all wavelengths are set to 860e-9 and all other
        hyperfine frequencies to zero to be adjusted.

        Returns
        -------
        list of pandas.DataFrame and pandas.Series entries.
            list of matrices specifying the frequencies of the participating
            transitions.
        """
        if len(self._freq) != 0:
            return self._freq
        out = dict2DF.get_DataFrame(self.const_dict,'HFfreq',gs_exs=self.label)
        if len(out) != 0:
            self._freq = out
        else:
            self._freq = self.DFzeros_without_mF()
        return self._freq
    
    def get_gfac(self):
        """Return a list of matrices for nicely displaying
        the g-factors of the ground states and the excited states respectively
        to conveniently specifying or modifying them. These g-factors are
        loaded from the .json file if available. Otherwise
        g-factors are set to 0 to be adjusted.
        
        Returns
        -------
        list of pandas.Series entries.
            list of two matrices for the g-factors of the ground and excited states.
        """
        if len(self._gfac) != 0:
            return self._gfac
        out = dict2DF.get_DataFrame(self.const_dict,'gfac',gs_exs=self.label)
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
        Name = {'gs':'ground','exs':'excited','ims':'intermediate','':''}[self.gs_exs]
        return "==> Electronic {} state {} with {:d} states in total".format(Name,self.label,self.N)
    
    def print_properties(self): 
        """Print all relevant constants and properties of the composed levelsystem
        in a convenient way to modify them if needed afterwards.
        """
        n=40
        print('\n{s:{c}^{n}}'.format(s=self.label,n=n,c='-'))
        print('\ng-factors:',   self.gfac, sep='\n')
        print('\nfrequencies (in MHz):', self.freq, sep='\n')
    
    def DFzeros_without_mF(self): #is this important anymore? is required for get functions when no const_dict is available
        """Return two arrays containing the labels of all levels without
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
        '''Return integer as number of defined :class:`State` instances.'''
        return len(self.states)

    @property
    def v_max(self):
        """Return the maximum quantum number ``v`` available in all state."""
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
        """Add a :class:`Lossstate` object to the self.entries
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
        """Delete a loss state if one is existing in the defined
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
        usage of function :meth:`~.Bfield.magn_remix`.
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
        #: decay rate :math:`\Gamma`
        if Gamma:
            self.Gamma = Gamma
        else:
            Gamma = dict2DF.get_DataFrame(self.const_dict,'Gamma',self.label)
            if len(Gamma) == 0:
                if self.verbose:
                    text = (f'Gamma must be defined for ElectronicState {self.label}!'
                            ' By default, it is now set to 1 MHz!')
                    warnings.warn(text)
                self.Gamma = 1.0 #1MHz
            else:
                self.Gamma = Gamma.iloc[0]
                
    def print_properties(self): 
        """Print all relevant constants and properties of the composed levelsystem
        in a convenient way to modify them if needed afterwards.
        """
        super().print_properties()
        n=40
        print('\nGamma (in MHz):\n{} {}'.format(self.gs_exs, self.label), self.Gamma)

class ElectronicImState(ElectronicGrState,ElectronicExState):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gs_exs = 'ims'
        
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
        """Return a deepcopy of this state's instance."""
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
        """Return `True` if a state is equal to an other state neglecting
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
    
    def check_QuNrvals(self,**QuNrvals):
        """Check if the State object possesses specific Quantum numbers with
        certain values.

        Parameters
        ----------
        **QuNrvals : kwargs
            Keyword arguments, e.g. v=0, F=1.

        Returns
        -------
        return_bool : bool
            True or False dependent on whether all Quantum numbers are included
            in the state.
        """
        return_bool = True
        for QuNr,val in QuNrvals.items():
            if QuNr not in self.QuNrs:
                return_bool = False
                break
            if self.__dict__[QuNr] != val:
                return_bool = False
                break
        return return_bool
    
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
        '''Return all the quantum numbers without mF'''
        return [QuNr for QuNr in self.QuNrs if QuNr != 'mF']
