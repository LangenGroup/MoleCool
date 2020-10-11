# -*- coding: utf-8 -*-
"""
Created on Thu May 14 02:03:38 2020

@author: fkogel

v2.1.0

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
import constants
import pandas as pd
from sympy.physics.wigner import wigner_3j,wigner_6j
#%%
class Levelsystem:  
    def __init__(self,load_constants='BaF'):
        self.grstates = Groundstates()
        self.exstates = Excitedstates(load_constants=load_constants)
        self.load_constants             = load_constants
        self.mass                       = constants.mass(name=load_constants)
        self._dMat                      = None
        self._dMat_red                  = None
        self._vibrbranch                = None
        self._freq                      = None
        self._gfac                      = None
        self.__dMat_arr                 = None
        self.__branratios_arr           = None
        self.__freq_arr                 = None
        self._muMat                     = None
        self._M_indices                 = None
        
    def __getitem__(self,index):
        #if indeces are integers or slices (e.g. obj[3] or obj[2:4])
        if isinstance(index, (int, slice)): 
            return self.entries[index]
        #if indices are tuples instead (e.g. obj[1,3,2])
        return [self.entries[i] for i in index]
    
    def __delitem__(self,index):
        del self.entries[index]
        self.__dMat_arr                 = None
        self.__branratios_arr           = None
        self.__freq_arr                 = None
        self._muMat                     = None
        self._M_indices                 = None
        
    def add_all_levels(self,nu_max):
        """
        Function to add all ground and excited states with a loss state
        in a convenient manner.

        Parameters
        ----------
        nu_max : int
            all ground states with vibrational levels :math:`\\nu\\le` `nu_max`
            and respectively all excited states up to `(nu_max-1)` are added to
            the subclasses :class:`Groundstates` and :class:`Excitedstates`.

        Returns
        -------
        None.
        """
        for nu in range(nu_max+1):
            self.grstates.add_grstate(nu=nu)      
            
        self.grstates.add_lossstate(nu=nu_max+1)
        
        if nu_max == 0: self.exstates.add_exstate(nu=0)
        else:
            for nu in range(nu_max):
                self.exstates.add_exstate(nu=nu)
    
    #%%
    def get_dMat(self,calculated_by = 'YanGroupnew'): #this function has to be resetted after a change of dMat_from which the new dMat should be calculated
        if np.all(self._dMat) != None: return self._dMat
        out = constants.dMat(name=self.load_constants)
        if out != None:
            self._dMat = pd.DataFrame(out[0],
                                     index  =pd.MultiIndex.from_arrays(out[1], names=('J','F','mF')),
                                     columns=pd.MultiIndex.from_arrays(out[2], names=("J'","F'","mF'"))
                                     )
        elif constants.dMat_red(name=self.load_constants) == None and constants.branratios(name=self.load_constants,calculated_by=calculated_by) != None:
            out = constants.branratios(name=self.load_constants,calculated_by=calculated_by)
            self._branratios = pd.DataFrame(out[0],
                                            index  =pd.MultiIndex.from_arrays(out[1], names=('J','F','mF')),
                                            columns=pd.MultiIndex.from_arrays(out[2], names=("J'","F'","mF'"))
                                            )
            print('WARNING: no dipole matrix or reduced dipole matrix found in constants.py, so the dipole matrix is constructed from the given branching ratios only with positive values!')
            self._dMat = self._branratios**0.5
        else:
            dMat_red = self.get_dMat_red()
            dMat = []
            row = 0
            column = 0
            index = []
            for J,F in np.array([dMat_red.index.get_level_values('J'), dMat_red.index.get_level_values('F')]).T:
                for mF in np.arange(-F,F+1):
                    dMat_row = []
                    index.append([J,F,mF])
                    columns = []
                    for J_,F_ in np.array([dMat_red.columns.get_level_values("J'"), dMat_red.columns.get_level_values("F'")]).T:
                        for mF_ in np.arange(-F_,F_+1):
                            #### or is it better in the other direction <ex|d|gr> ???
                            dMat_row.append( dMat_red.loc[(J,F),(J_,F_)] * (-1)**(F-mF)*np.sqrt(2*F+1) * float(wigner_3j(F,1,F_,-mF,mF-mF_,mF_)) )
                            columns.append([J_,F_,mF_])
                            column += 1
                    dMat.append(dMat_row)
                    row += 1
            self._dMat = pd.DataFrame(dMat,
                                     index  =pd.MultiIndex.from_arrays(np.array(index).T,   names=('J','F','mF')),
                                     columns=pd.MultiIndex.from_arrays(np.array(columns).T, names=("J'","F'","mF'"))
                                     )       
            self._dMat /= np.sqrt((self._dMat**2).sum(axis=0))
        return self._dMat
    
    def get_dMat_red(self):
        if np.all(self._dMat_red) != None: return self._dMat_red
        out = constants.dMat_red(name=self.load_constants)
        if out != None:
            self._dMat_red = pd.DataFrame(out[0],
                                     index  =pd.MultiIndex.from_arrays(out[1], names=('J','F')),
                                     columns=pd.MultiIndex.from_arrays(out[2], names=("J'","F'"))
                                     )
        elif constants.dMat(name=self.load_constants) != None:
            dMat = self.get_dMat().sort_index(axis='index')
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
            self._dMat = None
            rows, cols = self.QuNrs_without_mF
            self._dMat_red = pd.DataFrame(np.ones((rows.shape[-1],cols.shape[-1])),
                             index  =pd.MultiIndex.from_arrays(rows, names=('J','F')),
                             columns=pd.MultiIndex.from_arrays(cols, names=("J'","F'"))
                             )
            print('WARNING: there is no dipole matrix or reduced dipole matrix available! So a reduced matrix has been created only with ones:')
            print(self._dMat_red)
        return self._dMat_red

    def get_vibrbranch(self):
        if np.all(self._vibrbranch) != None: return self._vibrbranch
        out = constants.vibrbranch(name=self.load_constants)
        if out != None:
            self._vibrbranch = pd.DataFrame(out)
        else:
            self._vibrbranch = pd.DataFrame(np.ones((self.grstates.nu_max+1,self.exstates.nu_max+1))/(self.grstates.nu_max+1))
        self._vibrbranch.rename_axis("nu",axis=0,inplace=True)
        self._vibrbranch.rename_axis("nu'",axis=1,inplace=True)            
        return self._vibrbranch
        
    def get_freq(self):
        if self._freq != None: return self._freq
        out = constants.freq(name=self.load_constants)
        if out != None:
            self._freq = [pd.DataFrame(out[0]),
                          pd.Series(out[1][0], index=pd.MultiIndex.from_arrays(out[1][1], names=("J", "F"))),
                          pd.Series(out[2][0], index=pd.MultiIndex.from_arrays(out[2][1], names=("J'","F'")))]
        else:
            rows, cols = self.QuNrs_without_mF
            self._freq = [pd.DataFrame(860*np.ones((self.grstates.nu_max+1,self.exstates.nu_max+1))),
                          pd.Series(np.zeros(rows.shape[-1]),
                                    index=pd.MultiIndex.from_arrays(rows, names=('J','F'))),
                          pd.Series(np.zeros(cols.shape[-1]),
                                    index=pd.MultiIndex.from_arrays(cols, names=("J'","F'")))]
        self._freq[0].rename_axis("nu", axis=0,inplace=True)
        self._freq[0].rename_axis("nu'",axis=1,inplace=True)
        return self._freq
    
    def get_gfac(self):
        if self._gfac != None: return self._gfac
        out = constants.gfac(name=self.load_constants)
        if out != None:
            self._gfac = [pd.Series(out[0][0],
                                     index  =pd.MultiIndex.from_arrays(out[0][1], names=('J','F'))),
                         pd.Series(out[1][0],
                                     index  =pd.MultiIndex.from_arrays(out[1][1], names=("J'","F'")))
                                     ]
        else:
            rows, cols = self.QuNrs_without_mF
            self._gfac = [pd.Series(np.zeros(rows.shape[-1]),
                                     index=pd.MultiIndex.from_arrays(rows, names=('J','F'))),
                         pd.Series(np.zeros(cols.shape)[-1],
                                     index=pd.MultiIndex.from_arrays(cols, names=("J'","F'")))
                                     ]
        return self._gfac
    
    dMat        = property(get_dMat)
    dMat_red    = property(get_dMat_red)
    # def set_dMat_red(self,x):
    #     # self._dMat = None
    #     self._dMat_red = x
    # dMat_red = property(get_dMat_red,set_dMat_red)
    vibrbranch  = property(get_vibrbranch)
    freq        = property(get_freq)
    gfac        = property(get_gfac)
    #%%
    def calc_dMat(self,calculated_by='YanGroupNew'):
        #levels._dMat.xs((1.5,2,-1),level=('J','F','mF'),axis=0,drop_level=True).xs((0.5,1,-1),level=("J'","F'","mF'"),axis=1,drop_level=True)
        if np.all(self.__dMat_arr) != None: return self.__dMat_arr
        nu_max      = self.grstates.nu_max, self.exstates.nu_max
        vibrbranch  = self.get_vibrbranch().iloc[:nu_max[0]+1, :nu_max[1]+1]
        vibrbranch /= vibrbranch.sum(axis=0)
        dMat        = self.get_dMat(calculated_by=calculated_by)
        dMat       /= np.sqrt((dMat**2).sum(axis=0))
        self.__dMat_arr = np.zeros((self.lNum,self.uNum,3)) #is this normalization needed or only at the end of this function
        for l,gr in enumerate(self.grstates):
            for u,ex in enumerate(self.exstates):
                if gr.name == 'Loss state':
                    #the q=+-1,0 entries squared of the dMat are summed in the last line of the equations set in the Fokker-Planck paper.
                    # So for the loss state which should not interact with other levels, it doesn't matter which q component of the sum in the last line is contributing.
                    self.__dMat_arr[l,u,:] = np.array([1,0,0])*np.sqrt( vibrbranch.loc[gr.nu, ex.nu] )
                    continue
                pol = ex.mF-gr.mF
                if abs(pol) <= 1 and (gr.p+ex.p == 0):
                    self.__dMat_arr[l,u,int(pol)+1] = dMat.loc[(gr.J,gr.F,gr.mF),(ex.J,ex.F,ex.mF)]*np.sqrt(vibrbranch.loc[gr.nu, ex.nu])
        self.__dMat_arr /= np.sqrt((self.__dMat_arr**2).sum(axis=(2,0)))[None,:,None]
        np.nan_to_num(self.__dMat_arr,copy=False)
        return self.__dMat_arr
    
    def calc_branratios(self,calculated_by='YanGroupnew'):
        if np.all(self.__branratios_arr) != None: return self.__branratios_arr
        self.__branratios_arr = (self.calc_dMat(calculated_by=calculated_by)**2).sum(axis=2)
        return self.__branratios_arr
    
    def calc_freq(self):
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
        if np.all(self.__freq_arr) != None: return self.__freq_arr
        self.__freq_arr = np.zeros((self.lNum,self.uNum))
        lambda_vibr,hyperfine_gr,hyperfine_ex = self.get_freq()
        for l,gr in enumerate(self.grstates):
            for u,ex in enumerate(self.exstates):
                if gr.name == 'Loss state': # does this work???
                    self.__freq_arr[l,u] = 2*pi*(c/(lambda_vibr.loc[gr.nu,ex.nu]*1e-9) \
                                              - 0.0 + 1e6*hyperfine_ex.loc[(ex.J,ex.F)] )
                else:
                    self.__freq_arr[l,u] = 2*pi*(c/(lambda_vibr.loc[gr.nu,ex.nu]*1e-9) \
                                             - 1e6*hyperfine_gr.loc[(gr.J,gr.F)] + 1e6*hyperfine_ex.loc[(ex.J,ex.F)] )
        return self.__freq_arr    
    
    def calc_muMat(self):
        if self._muMat != None: return self._muMat
        # mu Matrix for magnetic remixing:
        # this matrix includes so far also off-diagonal non-zero elements (respective to F,F')
        # which will not be used in the OBEs calculation
        lNum, uNum = self.lNum, self.uNum
        self._muMat  = (np.zeros((lNum,lNum,3)), np.zeros((uNum,uNum,3)))
        gfac = self.get_gfac()
        for i0, states in enumerate([self.grstates,self.exstates]):
            for i1, st1 in enumerate(states):
                if st1.name == 'Loss state':
                    self._muMat[i0][i1,:,:] = np.zeros((lNum,3))
                    continue
                J,F,m = st1.J, st1.F, st1.mF
                for i2, st2 in enumerate(states):
                    if st2.name == 'Loss state':
                        self._muMat[i0][i1,i2,q+1] = 0.0
                        continue
                    n = st2.mF
                    for q in [-1,0,1]:
                        self._muMat[i0][i1,i2,q+1] = -gfac[i0].loc[(J,F)]* (-1)**(F-m)* \
                            np.sqrt(F*(F+1)*(2*F+1)) * float(wigner_3j(F,1,F,-m,q,n))
        return self._muMat
    
    def calc_M_indices(self):
        if self._M_indices != None: return self._M_indices
        # M indices for getting the indices when looping through all hyperfine
        # states which belong to a specific F or F' of the ground/ excited states
        M_indices_g,M_indices_e = [],[]
        for l1,st1 in enumerate(self.grstates):
            list_M = []
            if st1.name == 'Loss state':
                M_indices_g.append([l1])
                continue
            for l2,st2 in enumerate(self.grstates):
                if st2.name == 'Loss state':
                    continue
                if st1.nu == st2.nu and st1.N == st2.N \
                    and st1.J == st2.J and st1.F == st2.F:
                    list_M.append(l2)
            M_indices_g.append(list_M)
        for st1 in self.exstates:
            list_M = []
            for l2,st2 in enumerate(self.exstates):
                if st1.nu == st2.nu and st1.N == st2.N \
                    and st1.J == st2.J and st1.F == st2.F:
                    list_M.append(l2)
            M_indices_e.append(list_M)
        self._M_indices = (tuple(M_indices_g),tuple(M_indices_e))
        return self._M_indices
    
    #%%
    def __str__(self):
        #__str__ method is called when an object of a class is printed with print(obj)
        str(self.grstates)
        str(self.exstates)
        return self.description
    @property
    def print_properties(self):
        print('\ndipole matrix:',                 self.get_dMat(),
              '\nreduced dipole matrix',          self.get_dMat_red(),
              '\nvibrational branching:',         self.get_vibrbranch(),
              '\nfreq[0]: vibrational wavelengths (in nm):', self.get_freq()[0],
              '\nfreq[1]&[2]: hyperfine frequencies (in MHz)', self.get_freq()[1], self.get_freq()[2],
              '\ng-factors:',                     self.get_gfac()[0], self.get_gfac()[1],
              '\nGamma (in MHz):',                self.exstates.Gamma/2/pi*1e-6,
              '\nmass (in kg):',                          self.mass,
              sep='\n')
        return None
    @property
    def QuNrs_without_mF(self):
        row_labels, col_labels = [], []
        for l,gr in enumerate(self.grstates):
            if gr.name != 'Loss state':
                if not([gr.J,gr.F] in row_labels):
                    row_labels.append([gr.J,gr.F])
        for u,ex in enumerate(self.exstates):
            if not([ex.J,ex.F] in col_labels):
                col_labels.append([ex.J,ex.F])
        if len(row_labels) == 0 or len(col_labels)==0:
            raise('There are no levels defined! First, levels have to be added to this class')
        return np.array(row_labels).T, np.array(col_labels).T
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
        
    def add(self,nu=0,J=None,F=None,mF=None,N=1,S=.5,I=.5,p=None):
        #calculating all properties of the levels class does only work if all
        #levels are added first, and then the calculations are done afterwards.
        def isnumber(x):
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return True
            return False
        if isnumber(F) and isnumber(J):
            if isinstance(mF,(list,np.ndarray)):
                pass
            elif mF == None:
                mF = np.arange(-F,F+1)
            elif isnumber(mF):
                mF = [mF]
            else: print('WARNING: wrong datatype of mF')
            for mF_i in mF:
                if isnumber(p):
                    self.entries.append(Groundstate(nu,N,J,I,S,F,mF_i,p=p)) #I,S ???
                else:
                    self.entries.append(Groundstate(nu,N,J,I,S,F,mF_i,p=(-1)**N)) #I,S ???
        elif F==None and J==None:
            self.add_grstate(nu=nu,N=N,S=S,I=I) #parity p???
        else: print('Both J and F have to be specified or None of both.')
        
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
                    
    def add_lossstate(self,nu=None):
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
        if self.has_lossstate == False:
            if nu == None:
                self.entries.append(Lossstate(self.nu_max+1))
            else:
                self.entries.append(Lossstate(nu))
        else: print('loss state is already included')
    
    def del_lossstate(self):
        if self.has_lossstate == True:
            index = None
            for i,st in enumerate(self.entries):
                if st.name == 'Loss state':
                    index = i
            del self.entries[index]
        else: print('There is no loss state included to be deleted')
        
    def print_magn_remix(self):
        """Print out the magnetic remixing matrix of the ground states by the
        usage of function :func:`System.magn_remix`.
        """
        from System import magn_remix
        mat = magn_remix(self,0)
        for l1 in range(self.lNum):
            for l2 in range(self.lNum):
                if l2 == (self.lNum-1): end = '\n'
                else: 
                    end = ''
                if (l2+1) % 12 == 0: sep= '|'
                else: sep = ''     
                print(int(mat[l1,l2]),sep,end=end)
            if (l1 +1) % 12 == 0: print(self.lNum*2*'_')
        
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
    @property
    def has_lossstate(self):
        for gr in self.entries:
            if gr.name == 'Loss state': return True
        return False
    @property
    def nu_max(self):
        if len(self.entries)==0:
            raise('There are no levels defined! First, levels have to be added to this class')
        return max([st.nu for st in self.entries])
#%%    
class Excitedstates():
    def __init__(self,load_constants='BaF'):
        self.entries = []
        self.load_constants = load_constants
        #: decay rate :math:`\Gamma` which is received from the function
        #: :func:`BaFconstants.get_Gamma`
        self.Gamma = constants.Gamma(name=self.load_constants)

    def add(self,nu=0,J=None,F=None,mF=None,N=0,S=.5,I=.5,p=+1):
        def isnumber(x):
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return True
            return False
        if isnumber(F) and isnumber(J):
            if isinstance(mF,(list,np.ndarray)):
                pass
            elif mF == None:
                mF = np.arange(-F,F+1)
            elif isnumber(mF):
                mF = [mF]
            else: print('WARNING: wrong datatype of mF')
            for mF_i in mF:
                self.entries.append(Excitedstate(nu,N,J,I,S,F,mF_i,p=p)) #I,S ???
        elif F==None and J==None:
            self.add_exstate(nu=nu,N=N,S=S,I=I,p=p) #parity p???
        else: print('Both J and F have to be specified or None of both.')
        
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
    @property
    def nu_max(self):
        if len(self.entries)==0:
            raise('There are no levels defined! First, levels have to be added to this class')
        return max([st.nu for st in self.entries])
    
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
        
            