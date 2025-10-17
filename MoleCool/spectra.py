# -*- coding: utf-8 -*-
"""
Module for calculating the eigenenergies and eigenstates of diatomic molecules
exposed to external fields.
Therefore molecular constants which are measured and fitted in spectroscopic
experiments must be provided to build up the effective Hamiltonian terms.
Finally, the transition probabilities between two given electronic
state manifolds can be determined to simulate a complete spetrum.

Example
-------

Example for calculating and plotting a spectrum of 138BaF::
    
   from MoleCool.spectra import ElectronicStateConstants, Molecule, plt
   
   # defining spectroscopic constants
   const_gr = ElectronicStateConstants(const={
       'B_e' : 0.2159,   'D_e' : 1.85e-7,  'gamma' : 0.0027,
       'b_F' : 0.0022,   'c'   : 0.00027,
   })
   const_ex = ElectronicStateConstants(const={
       'B_e' : 0.2117,   'D_e' : 2.0e-7,   'A_e' : 632.2818,
       'p'   : -0.089545,'q'  : -0.0840,
       "g'_l": -0.536,   "g'_L":0.980,
       'T_e' : 11946.31676963,
   })

   # initiating empty Molecule instance and adding electronic states
   # with all quantum states up to a certain quantum number F
   BaF = Molecule(I1 = 0.5, mass = 157, temp = 4)
   BaF.add_electronicstate('X', 2, 'Sigma', const=const_gr)
   BaF.add_electronicstate('A', 2, 'Pi', Gamma=2.84, const=const_ex)
   BaF.build_states(Fmax=8)

   # calculating branching ratios and molecular spectrum
   BaF.calc_branratios()
   E, I = BaF.calc_spectrum(limits=(11627.0, 11632.8))
   
   # plotting
   plt.figure()
   plt.plot(E, I)
   plt.xlabel('Frequency (cm$^{-1}$)')
   plt.ylabel('Intensity (arb. u.)')
    
Tip
---
The instances of all classes :class:`Molecule`,
:class:`ElectronicState`, :class:`Hcasea` and :class:`ElectronicStateConstants`
can be printed::
    
    print(BaF)
    print(BaF.X)
    print(BaF.X.states[0])
    print(const_gr_138) # is the same as: print(BaF.X.const)
"""
from numba import jit
import numpy as np
import pandas as pd
from scipy.constants import c,h,hbar,pi,g,physical_constants
from scipy.constants import k as k_B
from scipy.constants import u as u_mass
from sympy.physics.wigner import clebsch_gordan,wigner_3j,wigner_6j
import json

from collections.abc import Iterable
from copy import deepcopy
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
#: Constant for converting a unit in wavenumbers (cm^-1) into MHz.
cm2MHz = 299792458.0*100*1e-6 #using scipys value for the speed of light
cm2THz = cm2MHz*1e-6

try:
    import pywigxjpf as wig
    def w3j(j_1, j_2, j_3, m_1, m_2, m_3):
        """returns Wigner 3j-symbol with arguments (j_1, j_2, j_3, m_1, m_2, m_3)"""
        return wig.wig3jj(int(2*j_1), int(2*j_2), int(2*j_3), int(2*m_1), int(2*m_2), int(2*m_3))
    def w6j(j_1, j_2, j_3, j_4, j_5, j_6):
        """returns Wigner 6j-symbol with arguments (j_1, j_2, j_3, j_4, j_5, j_6)"""
        return wig.wig6jj(int(2*j_1), int(2*j_2), int(2*j_3), int(2*j_4), int(2*j_5), int(2*j_6))
    max_two_j = 2*120
    # wig.wig_table_init(max_two_j,3)
    wig.wig_table_init(max_two_j,6)
    wig.wig_temp_init(max_two_j)
except ModuleNotFoundError:
    def w3j(j_1, j_2, j_3, m_1, m_2, m_3):
        """returns Wigner 3j-symbol with arguments (j_1, j_2, j_3, m_1, m_2, m_3)"""
        return float(wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3))
    def w6j(j_1, j_2, j_3, j_4, j_5, j_6):
        """returns Wigner 6j-symbol with arguments (j_1, j_2, j_3, j_4, j_5, j_6)"""
        return float(wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6))
#%% classes
class Molecule:
    def __init__(self,I1=0,I2=0,Bfield=0.0,mass=0,load_constants=None,
                 temp=5.0,naturalabund=1.0,label='BaF',verbose=True,mF_states=False):        
        """This class represents a molecule containing all electronic and
        hyperfine states in order to calculate branching ratios and thus
        to plot the spectrum.

        Parameters
        ----------
        I1 : float, optional
            nuclear spin of one atom of the molecule. The default is 0.
        I2 : float, optional
            If the first nuclear spin `I1` is non-zero, a second smaller
            nuclear spin can be provided via `I2`. The default is 0.
        Bfield : float, optional
            strength of a external magnetic field in T. The default is 0.0.
        mass : float, optional
            mass of the molecule in atomic mass units. The default is None.
        load_constants : string, optional
            if provided the molecular constants are loaded from an external
            file. The default is None.
        temp : float, optional
            temperatur of the molecule in K. The default is 5.0.
        naturalabund : float, optional
            natural abundance in the range from 0.0 to 1.0 of different isotopes
            of a molecule in order to weight the spectra for different isotopes.
            The default is 1.0.
        label : string, optional
            label or name of the molecule. The default is 'BaF'.
        verbose : bool, optional
            specifies if additional information and warning should be printed
            during the calculations. The default is True.
        """
        self.I1         = I1
        self.I2         = I2
        self.Bfield     = Bfield
        self.mass       = mass
        self.temp       = temp
        self.naturalabund = naturalabund
        self.label      = label
        self.verbose    = verbose
        self.mF_states  = bool(mF_states)
        self.grstates   = [] #empty lists to hold labels of electronic states
        self.exstates   = [] #to be appended later
        if load_constants: #or better in Electronic state???
            #maybe call other function to import molecular constants from a file
            pass
        #elements, isotopes, natural abundance, nuclear spin magn moment, ..
        
        try: #only works if module could be imported
            max_two_j = 2*80
            # wig.wig_table_init(max_two_j,3)
            wig.wig_table_init(max_two_j,6)
            wig.wig_temp_init(max_two_j)
        except NameError:
            pass
        
    def add_electronicstate(self,*args,**kwargs):
        """adds an electronic state (ground or excited state) as instance of
        the class :class:`ElectronicState` to this :class:`~Molecule.Molecule`.
        
        Parameters
        ----------
        args, kwargs
            arguments and keyword arguments for the eletronic state (see
            :class:`ElectronicState` for the required arguments)
        """
        self.__dict__[args[0]] = ElectronicState(*args,**kwargs)
        self.__dict__[args[0]].I1 = self.I1
        self.__dict__[args[0]].I2 = self.I2
        self.__dict__[args[0]].Bfield = self.Bfield
        self.__dict__[args[0]].mF_states = self.mF_states
        self.__dict__[args[0]].verbose = self.verbose
        if self.__dict__[args[0]].grex == 'ground state':
            self.grstates.append(args[0])
        else: self.exstates.append(args[0])
        
    def build_states(self,Fmax,Fmin=None):
        """Builds the individual Quantum states within all electronic states
        defined within the Molecule's instance in the range from Fmin to Fmax
        in units of the total angular momentum quantum number.
        See :meth:`ElectronicState.build_states` for details.

        Parameters
        ----------
        Fmax : float
            maximum angular momentum quantum number.
        Fmin : float, optional
            minimum angular momentum quantum number. The default is None.
        """
        for ElSt in [*self.grstates, *self.exstates]:
            self.__dict__[ElSt].build_states(Fmax=Fmax, Fmin=Fmin)
    
    def calc_branratios(self,threshold=0.0,include_Boltzmann=True,
                        grstate=None,exstate=None):
        """
        calculates the linestrengths (by evaluating the electric dipole matrix)
        and energies of all transitions between a ground and excited electronic
        state in order to obtain the branching ratios weighted by a Boltzmann
        factor.
        
        Note
        ----
        This method creates the attributes (as `numpy.ndarrays`):
        
        * ``dipmat`` : electric dipole matrix of the eigenstates in the same
          order as the eigenstates are stored in the respective electronic
          states :class:`ElectronicState` which can be printed via the method
          :meth:`~ElectronicState.get_eigenstates`.
        * ``E`` : transition energies between the eigenstates in the same order
        * ``branratios`` : respective branching ratios

        Parameters
        ----------
        threshold : float, optional
            all branching ratios below the threshold in the range from 0.0
            to 1.0 are set to zero. The default is 0.0.
        include_Boltzmann : bool, optional
            determines if the Boltzmann factor is included weighting ground state
            levels with different energy dependent on the temperature.
            The default is True.
        grstate : string, optional
            label of the electronic excited state which should be used for the
            calculation of the branching ratios. By default the last added
            ground state is used. The default is None.
        exstate : string, optional
            label of the electronic excited state which should be used for the
            calculation of the branching ratios. By default the last added
            excited state is used. The default is None.
        """
        # if grstate is provided use this string as label of the ground state
        # otherwise use the last added ground state within the Molecule class
        # -> same for the excited state whose variable name is A here 
        if grstate == None: grstate = self.grstates[-1]
        if exstate == None: exstate = self.exstates[-1]
        X, A = self.__dict__[grstate], self.__dict__[exstate]
        self.branratios_labels = (grstate,exstate)
        
        #hamiltonian matrix elements (of the el. dipole Operator) of the pure basis states
        if self.verbose: print('Calculating linestrengths in pure basis')
        H_dpure = np.zeros((X.N,A.N))
        for ii,pure_l in enumerate(X.states):
            for jj,pure_u in enumerate(A.states):
                H_dpure[ii,jj] = H_d(pure_l,pure_u)
        
        # if eigenstates are not calculated so far, do this
        if not X.eigenst_already_calc: X.calc_eigenstates()
        if not A.eigenst_already_calc: A.calc_eigenstates()
        
        # eigenstate dipole matrix via matrix multiplication of eigenstates and pure basis dipole matrix 
        self.dipmat = np.matmul(np.matmul(X.Ev.T,H_dpure),A.Ev)
        
        # transition frequency as energy offset
        E_offset = A.const.electrovibr_energy - X.const.electrovibr_energy
        self.E = A.Ew[None,:] - X.Ew[:,None] + E_offset
        
        if include_Boltzmann and (self.temp>0.0):
            fac         = 1*cm2MHz*1e6*h/k_B
            E_lowest    = np.min(X.Ew)
            Boltzmannfac= np.exp(-(X.Ew-E_lowest)[:,None]/fac/self.temp)
            Boltzmannfac/= Boltzmannfac[:,0].sum() #normalizing constant in Boltzmann statistic
        else:
            Boltzmannfac = 1.0
        self.branratios = self.dipmat**2 * Boltzmannfac
        # additional degeneracy factor here?
        
        #set all branching ratios smaller than the threshold to zero
        self.branratios = np.where(self.branratios<=threshold*np.max(self.branratios),0.0,self.branratios)
    
    def calc_spectrum(self,limits=[],sigma=None,plotpoints=40000):
        """ calculates the spectrum in a certain frequency range using the
        branching ratios previously calculated in the method :func:`calc_branratios`.
        The resulting frequency and intensity arrays are not only returned but
        also stored as variables ``Eplt`` and ``I`` in the :class:`~Molecule.Molecule`
        instance. The widths of the single transitions are determined by the
        natural linewidth ``Gamma`` of the respective :class:`~Molecule.ElectronicState`
        instance (Lorentzian profile) and the temperature (Gaussian profile).
        The convolution of both profiles is then given by the Voigt profile.

        Parameters
        ----------
        limits : list, optional
            defines the frequency limits for the plotting the spectrum as list
            or tuple of size 2 in units of wavenumbers 1/cm. By default the
            complete range containing all transitions is chosen. 
            The default is None.
        sigma : float, optional
            if desired, one can manually define the width of the Doppler-broadening,
            which would actually arise due to a non-zero temperature (by default),
            to a specific value (in cm^-1). The default is None.
        plotpoints : int, optional
            integer number specifying the number of intervals for the plotting
            frequency range, i.e. the plot resolution. The default is 40000.
            
        Returns
        -------
        numpy.ndarray
            frequency array of the spectrum to be plotted.
        numpy.ndarray
            intensity array of the spectrum belonging to the frequency array.
        """
        grstate, exstate = self.branratios_labels
        X, A        = self.__dict__[grstate], self.__dict__[exstate]
        Gamma       = A.Gamma
        if len(limits) == 0:
            Emin,Emax   = np.min(self.E)-0.1 , np.max(self.E)+0.1
        else:
            Emin,Emax = limits
        Eplt        = np.linspace(Emin,Emax,plotpoints)
        I           = np.zeros(Eplt.size)
        from scipy.special import voigt_profile
        if sigma == None:
            sigma = (Emax+Emin)/2 *np.sqrt(8*k_B*self.temp*np.log(2)/(self.mass*u_mass*c**2))
        for i in range(X.N):
            for j in range(A.N):
                branratio = self.branratios[i,j]
                if branratio == 0.0: continue
                I += branratio * voigt_profile(self.E[i,j]-Eplt,sigma,Gamma/2)
        I *= self.naturalabund
        self.Eplt,self.I = Eplt,I
        return self.Eplt, self.I
    
    def which_eigenstates(self,Emin,Emax):
        """
        searches all eigenstates which are part of the transitions
        within the specified frequency range. These eigenstates are printed
        with their branching ratios and transitionenergies.

        Parameters
        ----------
        Emin : float
            lower limit of the frequency range.
        Emax : float
            upper limit of the frequency range.
        """
        st_l_arr, st_u_arr = np.where( (self.E > Emin) & (self.E < Emax) & (self.branratios > 0.0))
        for i in range(st_l_arr.size):
            st_l,st_u = st_l_arr[i], st_u_arr[i]
            print('lower eigenstate {:3} & upper eigenstate {:3}, branratio{:5.1f}%, energy {:8f} THz'.format(
                st_l,st_u,self.branratios[st_l,st_u]*100,self.E[st_l,st_u]*cm2MHz*1e-6))
        # for st_l in np.unique(st_l_arr):
        #     print(self.X.Ev[:,st_l])
        
    def get_dMat_red(self,recalc_branratios=True, Hcasebasis=True,
                     onlygoodQuNrs=True, index_filter=None, **kwargs):
        """Outputs the reduced electric dipole matrix in a nice readable format.

        Parameters
        ----------
        recalc_branratios : bool, optional
            Whether the branching ratios should be calculated again.
            The default is True.
        Hcasebasis : bool, optional
            See :meth:`ElectronicState.get_eigenstates`. The default is True.
        onlygoodQuNrs : bool, optional
            See :meth:`ElectronicState.get_eigenstates`. The default is True.
        index_filter : tuple of dict
            Tuple or list including two dictionaries, i.e. see two arguments of
            :func:`multiindex_filter`. Default is no filtering.
        **kwargs : kwargs
            Additional keyword arguments (see :meth:`calc_branratios`). By default
            these keyword arguments are ``dict(threshold=0.0, include_Boltzmann=False)``.

        Returns
        -------
        pandas.DataFrame
            reduced electric dipole matrix.
        """
        kwargs_calc_branratios = dict(dict(threshold=0.0,include_Boltzmann=False),
                                      **kwargs)
        if ('dipmat' not in self.__dict__.keys()) or recalc_branratios:
            self.calc_branratios(**kwargs_calc_branratios)
        GrSt, ExSt = [self.__dict__[lab] for lab in self.branratios_labels]
        Eigenbasis = [ElSt.get_eigenstates(Hcasebasis=Hcasebasis,
                                           onlygoodQuNrs=onlygoodQuNrs,
                                           mixed_states=False)
                      for ElSt in [GrSt,ExSt]]
        DF = pd.DataFrame(self.dipmat,
                          index=Eigenbasis[0].index, columns=Eigenbasis[1].index)
        if index_filter:
            return multiindex_filter(DF, rows=index_filter[0], cols=index_filter[1], drop_level=False)
        else:
            return DF
    
    def get_E(self, index_filter=None, **kwargs):
        """Similar method as :meth:`get_dMat_red` but with the eigenenergies.

        Parameters
        ----------
        index_filter : tuple of dict
            Tuple or list including two dictionaries, i.e. see two arguments of
            :func:`multiindex_filter`. Default is no filtering.
        **kwargs : kwargs
            keyword arguments from :meth:`get_dMat_red`.

        Returns
        -------
        pandas.DataFrame
            eigen energies.
        """
        DF = self.get_dMat_red(**kwargs)
        DF.iloc[:,:] = self.E
        if index_filter:
            return multiindex_filter(DF, rows=index_filter[0], cols=index_filter[1], drop_level=False)
        else:
            return DF
    
    def get_branratios(self,normalize=True,include_Boltzmann=False,threshold=0.0,
                       index_filter=None,**kwargs):
        """Similar method as :meth:`get_dMat_red` but with the branching ratios.

        Parameters
        ----------
        normalize : bool, optional
            Whether the columns of the branching ratios should be normalized to 1.
            The default is True.
        include_Boltzmann : bool, optional
            See :meth:`get_dMat_red`. The default is False.
        threshold : float, optional
            See :meth:`get_dMat_red`. The default is 0.0.
        index_filter : tuple of dict
            Tuple or list including two dictionaries, i.e. see two arguments of
            :func:`multiindex_filter`. Default is no filtering.
        **kwargs : kwargs
            Additional keyword arguments (see :meth:`get_dMat_red`).

        Returns
        -------
        pandas.DataFrame
            branching ratios.
        """
        DF = self.get_dMat_red(include_Boltzmann=include_Boltzmann,
                               threshold=threshold,**kwargs)
        DF.iloc[:,:] = self.branratios
        if normalize:    
            DF /= DF.sum(axis=0)
        if index_filter:
            return multiindex_filter(DF, rows=index_filter[0], cols=index_filter[1], drop_level=False)
        else:
            return DF
    
    def export_OBE_properties(self, gs=None, exs=None, index_filter=({},{}),
                              fname = '', HFfreq_offsets=[0,0], Bmaxs=[1e-4,1e-4],
                              QuNrs_const = ([],[]), QuNrs_var = ([],[]),
                              include_mF=False, vibr_values={}, rounded=None):
        """Export all the important properties connected to two electronic
        states to a dictionary in a proper format for the OBE simulation code to
        import the properties. This dictionary can also directly be saved as
        a .json file. This method uses the similar function
        :meth:`ElectronicState.export_OBE_properties`.

        Parameters
        ----------
        gs : str, optional
            label of the ground ElectronicState. The default is None meaning that
            it is automatically chosen.
        exs : str, optional
            label of the excited ElectronicState. The default is None meaning that
            it is automatically chosen.
        index_filter : tuple(dict), optional
            to filter the states of interest. The tuple with two dictionaries
            is used for ground and excited state see :func:`multiindex_filter`.
            The default is ({},{}).
        fname : str, optional
            filename of the constants file. Should end with '.json'.
            The default is '' meaning that no file will be saved.
        HFfreq_offsets : list(float), optional
            offsets for the hyperfine frequencies of the two electronic states.
            See `HFfreq_offset` in :meth:`ElectronicState.export_OBE_properties`.
            The default is [0,0].
        Bmaxs : list(float), optional
            Determines how the gfactors of the two ElectronicStates are calculated 
            (see :meth:`ElectronicState.get_gfactors`). The default is [1e-4,1e-4].
        QuNrs_const : tuple(list), optional
            constant Quantum numbers (see :func:`get_QuNr_keyval_pairs`).
            The default is ([],[]).
        QuNrs_var : TYPE, optional
            Variable Quantum numbers (see :meth:`ElectronicState.export_OBE_properties`).
            The default is ([],[]).
        include_mF : bool, optional
            Whether to include mF Quantum numbers on top of the ones from `QuNrs_var`.
            The default is False.
        vibr_values : dict, optional
            Dictionary to include the vibrational properties manually.
            Includes keys 'vibrbranch', 'vibrfreq', 'QuNrs_rows', 'QuNrs_cols'
            where the last two keys are optional as they will be generated
            automatically. The default is {}.
        rounded : int, optional
            digit to round the frequencies and gfactors. The default is None
            meaning that no rounding is applied.

        Returns
        -------
        dict
            Dictionary with the properly formatted level and transition properties.
        """
        # checking QuNrs_const and QuNrs_var
        if len(QuNrs_const[0]) != len(QuNrs_const[1]):
            raise Exception((f"Length of both QuNrs_const list items {QuNrs_const[0]}"
                             f" and {QuNrs_const[1]} must be the same."))
        for QuNrs_var_i in QuNrs_var:
            if 'mF' in QuNrs_var_i:
                QuNrs_var_i.remove('mF')
                
        # checking vibrational values
        if ('vibrbranch' in vibr_values) and ('vibrfreq' in vibr_values):
            for i, key in enumerate(['QuNrs_rows','QuNrs_cols']):
                if key not in vibr_values:
                    vibr_values.update({key : dict(
                        v=list(range(np.array(vibr_values['vibrfreq']).shape[i])))})
        else:
            vibr_values = {}
        
        # checking ground and excited electronic state
        if not gs:  gs  = self.grstates[0]
        if not exs: exs = self.exstates[0]
        GrState     = self.__dict__[gs]
        ExState     = self.__dict__[exs]
        
        ##### creating very large dictionary that is saved as json in the end #####
        dic0        = NestedDict()
        dic0.update({'mass' : self.mass, 'level-specific' : dict()})
        
        ##### transition-specific #####
        for GrExState in [GrState, ExState]:
            GrExState.mF_states = bool(include_mF)
            GrExState.Bfield    = 0
            GrExState.build_states(GrExState.Fmax)
        
        # calculate and round dipole matrix
        DF          = self.get_dMat_red(recalc_branratios=True, Hcasebasis=True,
                                        index_filter=index_filter)
        if rounded != None:
            DF = DF.round(rounded)
        
        # putting the data into the large dictionary:
        # dic is a sub dict with only the dMat data and the respective rows and cols QuNrs
        dic     = {['dMat_red','dMat'][int(bool(include_mF))] : DF.to_numpy().tolist()}
        for key, MI, QuNrs_var_i in zip(['QuNrs_rows','QuNrs_cols'], [DF.index, DF.columns], QuNrs_var):
            if not QuNrs_var_i:
                QuNrs_var_ = get_unique_multiindex_names(MI)
            else:
                QuNrs_var_ = QuNrs_var_i.copy()
                if include_mF:
                    QuNrs_var_.append('mF')
            dic[key] = get_QuNrvals_from_multiindex(MI, QuNrs_var_)

        key_list = ['transition-specific',
                    *get_QuNr_keyval_pairs([self.X,self.A], [DF.index, DF.columns], QuNrs_const)]
        dic0[key_list] = dic
        dic0[key_list[:2]].update(vibr_values)
        
        ##### level-specific #####
        for i, ElState in enumerate([GrState,ExState]):
            dic0['level-specific'].update(
                ElState.export_OBE_properties(
                    index_filter=index_filter[i], rounded=rounded, nested_dict=True,
                    QuNrs=QuNrs_var[i].copy(), HFfreq_offset=HFfreq_offsets[i], Bmax=Bmaxs[i],
                    get_QuNr_keyval_pairs_kwargs=dict(include_v=True,QuNrs_names=QuNrs_const[i]))
                )
        
        # saving as json file
        if fname and isinstance(fname,str):
            with open(fname, 'w') as file:
                json.dump(dic0.copy(), file, sort_keys=False, indent=4)
        
        return dic0.copy()
    
    def plot_fortrat(self, QuNrs, ax=None, limits=None, branratio_TH=1e-2,
                     limits_unit='THz', markers = ['+','x','.','1','2','3','4'],
                     legend=True, xaxis_func=lambda x: x):
        """Plotting Quantum number values over transition frequencies. It
        makes sense to combine this plot with the actual spectrum.

        Parameters
        ----------
        QuNrs : list
            Quantum number names for which the values are plotted.
        ax : matplotlib.axis, optional
            axis where to put the plot. The default is None meaning to use
            ``plt.gca()``.
        limits : tuple or list, optional
            frequency limits which determine the minimum and maximum of the 
            frequency axis. The default is None meaning the whole range is used.
        branratio_TH : float, optional
            branching ratio threshold. Transitions with smaller branching ratios
            are ignored. The default is 1e-2.
        limits_unit : str, optional
            Unit of the limits that are provided (['THz','cm-1','MHz']).
            The default is 'THz'.
        markers : list(str), optional
            markers to be used for the plotted points for each Quantum number.
            The default is ['+','x','.','1','2','3','4'].
        legend : bool, optional
            Whether to use a legend. The default is True.
        xaxis_func : func, optional
            function to convert the x axis. The default is lambda x: x.
        """
        if not ax:
            ax = plt.gca()
            
        E       = self.get_E()
        brans   = self.get_branratios().to_numpy()

        E_np    = E.to_numpy()
        if not np.all(limits):
            limits = (self.E.min(), self.E.max())
        else:
            limits = np.array(limits) * {'THz':1/cm2THz, 'cm-1':1, 'MHz':1/cm2MHz}[limits_unit]
            
        inds    = np.argwhere((E_np>limits[0]) & (E_np<limits[1]) & (brans>branratio_TH))
        E_index = E.index.to_frame(index=False) # ground state QuNrs
        E_cols  = E.columns.to_frame(index=False) # excited state QuNrs
        
        for i,QuNr in enumerate(QuNrs):
            if QuNr[-1] == "'":
                QuNrs_vals = E_cols[QuNr[:-1]].iloc[inds[:,1]] # excited state
            else:
                QuNrs_vals = E_index[QuNr].iloc[inds[:,0]] # ground state
                
            E_arr   = np.array([E_np[ixy[0],ixy[1]] for ixy in inds]) # eigenenergies
            ax.plot(xaxis_func(E_arr), QuNrs_vals, marker=markers[i], ls='', label=QuNr)
            ax.set_ylabel('Quant. Nrs.')
            
        if legend:
            ax.legend()
    
    def __str__(self):
        """prints all general information of the Molecule with its electronic states"""
        
        str1 = 'Molecule {}: with the nuclear spins I1 = {}, I2 = {} and mass {} u'.format(
            self.label,self.I1,self.I2,self.mass)
        str1+= '\n magnetic field strength: {:.2e}G, temperature: {:.2f}K'.format(
            self.Bfield*1e4,self.temp)
        str1+= '\nIncluding the following defined electronic states:'
        for state in [*self.grstates,*self.exstates]:
            str1+='\n\n* {}'.format(self.__dict__[state])
        
        return str1
#%%
class ElectronicStateConstants:
    #: electronic energy offset constant (this constant equals the transition frequency if no vibrational constants are given. (see :meth:`ElectronicStateConstants.electrovibr_energy`).) 
    const_elec      = ['T_e']
    #: vibrational constants
    const_vib       = ['w_e','w_e x_e','w_e y_e']
    #: rotation constants
    const_rot       = ['B_e','D_e','H_e','alpha_e','gamma_e','beta_e']
    #: spin-rotation constants
    const_sr        = ['gamma','gamma_D']
    #: spin-orbit constants
    const_so        = ['A_e','alpha_A','A_D']
    #: hyperfine constants
    const_HFS       = ['a','b_F','c','d','c_I',
                       'a_2','b_F_2','c_2','d_2'] #for the second nuclear spin if I2 is non-zero
    #: electric quadrupol interaction
    const_eq0Q      = ['eq0Q']
    #: Lambda-doubling constants
    const_LD        = ['o','p','q']#,'p_D','q_D'] can maybe extracted out of Fortran code
    #: (magnetic) Zeeman constants. `g_S` and `g'_L` are initially set to 2.002 and 1. respectively.
    const_Zeeman    = ['g_l',"g'_l",'g_S',"g'_L"]  
    #: sum of all constant names
    const_all = const_elec + const_vib + const_rot + const_sr + const_so \
                + const_HFS + const_eq0Q + const_LD + const_Zeeman
    #: dictionary of all predefined constants which are set to zero initially
    constants_zero = dict.fromkeys(const_all, 0.0)
    # standard Zeeman Hamiltonian constants
    constants_zero['g_S'] = 2.002#3 in Fortran code without the last digit?
    constants_zero["g'_L"] = 1.0

    def __init__(self,const={},unit='1/cm',nu=0,**kwargs):
        """An instance of this class represents an object which includes all
        molecular constants for evaluating the effective Hamiltonian yielding
        the molecular eigenstates and respective eigenenergies.
   
        After the provided constants are loaded into the instance they can simply
        be modified or returned via::
            
            const = ElectronicStateConstants()
            const['B_e'] = 0.21
            print(const['g_S'])

        Parameters
        ----------
        const : dict, optional
            dictionary of all constants in wave numbers (1/cm) required for the
            effective Hamiltonian.
            See the predefined constant attributes of this class,
            e.g. :attr:`const_vib` or :attr:`const_rot`, containing all possible
            names of the constants which are set to zero initially.
            The values of the provided dictionary `const` are then
            loaded into the class' new instance. The default is {}.
        unit : str, optional
            unit of the provided constants. Can either be '1/cm' for wave
            numbers or 'MHz' for frequency. The default is '1/cm'.
        nu : int, optional
            vibrational quantum number for the vibrational levels.
            The default is 0.
        **kwargs : optional
            the values of the provided dictionary `const` can also be given as
            normal keyword arguments, e.g. B_e = 0.21, which will overwrite
            the ones from the dictionary.

        Tip
        ---
        Such instance can nicely export the defined constants as a HTML file
        (see :meth:`show`) or can be saved with all its properties
        via the function :func:`~.tools.save_object` and can be loaded
        later using the function :func:`~.tools.open_object`.
        
        Raises
        ------
        KeyError
            if the dictionary `const` or the keyword arguments `kwargs`
            contains some values for which the respective key is not defined.
        """
        units = ['1/cm','MHz']
        # load all constants from constants_zero into the class' instance
        # & update non-zero values from the const dictionary
        self.__dict__.update(self.constants_zero.copy())
        const.update(kwargs) # merge provided **kwargs with the const dictionary
        if not (unit in units):
            raise ValueError('{} is not a valid unit. Use instead one of: {}'.format(unit,units))
        for key,value in const.items():
            if not (key in self.const_all):
                raise KeyError("key '{}' of the input parameter 'const' does not exist".format(key))
            if (unit == 'MHz') and (not (key in self.const_Zeeman)):
                const[key] = value/cm2MHz
        self.__dict__.update(const)
        self.nu = nu
        # not really a constant but it is needed in this class for calculation
        # of some vibrationally dependent constants
        
    def show(self,formatting='all',createHTML=False):
        """returns a `pandas.DataFrame` object which shows the defined constants
        in a nice format. This table can then be saved as a `.html` file.

        Parameters
        ----------
        formatting : str, optional
            Can either be 'all' for printing all constants or 'non-zero' for
            printing only the non-zero constants. The default is 'all'.
        createHTML : bool or str, optional
            if True the returned table with the specific formatting is saved
            as a HTML file `ElectronicStateConstants.html`. If str the HTML file
            is saved with this filename. The default is False.

        Returns
        -------
        DF : pandas.DataFrame
            table of all constants with further explanatory comments.
        """
        names = ['electronic energy offset','vibration','rotation','spin-rotation','spin-orbit',
                 'hyperfine','electric quadrupol','Lambda-doubling','Zeeman (no unit)']
        const_vars = ['const_elec','const_vib','const_rot','const_sr','const_so',
                      'const_HFS','const_eq0Q','const_LD','const_Zeeman']
        index, values, values2  = [],[],[]
        precision = 9
        precision_old = pd.get_option('display.precision')
        pd.set_option("display.precision", precision)
        for arr,name in zip(const_vars,names):
            for var in self.__class__.__dict__[arr]:
                index.append([name,var])
                values.append(self.__dict__[var])
                if arr == 'const_Zeeman':
                    values2.append(self.__dict__[var])
                else:
                    values2.append(self.__dict__[var]*cm2MHz)
        value_arr = np.array([values,values2])
        DF = pd.DataFrame(value_arr.T,
                          index=pd.MultiIndex.from_arrays(np.array(index).transpose()),
                          columns=['value (1/cm)','value (MHz)'])
        
        if formatting == 'all':
            pass    
        elif formatting == 'non-zero':
            indices = np.where(np.array(values) != 0.0)[0]
            DF = DF.iloc[indices]
        
        if createHTML:
            #render dataframe as html
            html = DF.to_html(formatters=('{:.6f}'.format,'{:.2f}'.format),justify='right')
            #write html to file
            if type(createHTML) == str:
                text_file = open(createHTML+'.html', "w")
            else:
                text_file = open("ElectronicStateConstants.html", "w")
            text_file.write(html)
            text_file.close()
            
        pd.set_option("display.precision", precision_old)
        return DF
    
    def get_isotope_shifted_constants(self,masses,masses_isotope,
                                      g_N=0,g_N_isotope=0,inplace=False):
        """calculates new isotope-shifted constant set.
        --> see https://www.lfd.uci.edu/~gohlke/molmass/ for precise masses.
        
        Parameters
        ----------
        masses : list or ndarray
            masses of the two atoms of the molecule for which the constants
            are given, e.g. [137.9052, 18.9984] for 138BaF.
        masses_isotope : list or ndarray
            masses of the two atoms of the molecule for which the constants
            should be calculated
        inplace : bool
            determines if the current constants are replaced inplace.
            
        Returns
        -------
        ElectronicStateConstants
            Copy of the current ElectronicStateConstants instance with
            isotope-shifted constants.
        """
        masses,masses_isotope = np.array(masses), np.array(masses_isotope)
        if (len(masses) != 2) or (len(masses_isotope) != 2):
            raise ValueError('masses parameters must be of length 2')
        
        # reduced masses ratios rho and rho_el
        m_e     = 1/1836.15267343
        def m_mol(masses):
            return masses[0]*masses[1]/masses.sum()
        def m_el(masses):
            return m_e*(masses.sum()-m_e) / masses.sum()
        rho     = np.sqrt( m_mol(masses)/m_mol(masses_isotope) )
        rho_el  = m_el(masses) / m_el(masses_isotope)
        
        # nuclear magnetic moment ratio (nuclear g-factor ratio with spin numbers)
        if g_N != 0:
            g_N_ratio = g_N_isotope / g_N
        else:
            g_N_ratio = 0
        
        if inplace:
            const_new = self
        else:
            const_new = deepcopy(self)
        
        # others
        const_new['gamma']      = self['gamma']*rho**2
        const_new['p']          = self['p']*rho**2
        const_new['q']          = self['q']*rho**4
        # spin-orbit
        # A_e has no isotopic shift (doi:10.1006/jmsp.2000.8252)
        const_new['alpha_A']    = self['alpha_A']*rho
        const_new['A_D']        = self['A_D']*rho**2
        # rotation
        const_new['B_e']        = self['B_e']*rho**2        # Y_01 Dunham coeffs.
        const_new['alpha_e']    = self['alpha_e']*rho**3    #-Y_11
        const_new['gamma_e']    = self['gamma_e']*rho**4    #-Y_21
        const_new['D_e']        = self['D_e']*rho**4        # Y_02
        const_new['beta_e']     = self['beta_e']*rho**5     # Y_12
        # vibration
        const_new['w_e']        = self['w_e'] * rho
        const_new['w_e x_e']    = self['w_e x_e'] * rho**2
        const_new['w_e y_e']    = self['w_e y_e'] * rho**3
        const_new['T_e']        = self['T_e'] / rho_el
        #nuclear spin --> these constants must belong to the correct atom in the molecule!?
        # maybe define atom1 and atom2 in the Molecule instance with respective g_N factors
        const_new['a']          = self['a'] * g_N_ratio
        const_new['b_F']        = self['b_F'] * g_N_ratio
        const_new['c']          = self['c'] * g_N_ratio
        const_new['d']          = self['d'] * g_N_ratio

        return const_new
    
    def to_dict(self,include_vdep_consts=True,exclude_default=False):
        """Converts the defined constants to a dictionary which can also include
        calculated values, e.g. :meth:`B_v` and :meth:`D_v`.
        
        Parameters
        ----------
        include_vdep_consts : bool, optional
            whether the dictionary includes also the calculated values, e.g.
            :meth:`B_v`, :meth:`D_v`, and :meth:`A_v`. The default is True.
        exclude_default : bool, optional
            whether the dictionary includes all possible constants (False) or only
            the constants which differ from the default values (True).
            The default is False.
            
        Returns
        -------
        dic : dict
            dictionary with all defined constants.
        """
        dic = {key : self.__dict__[key] for key in self.const_all}
        
        if exclude_default:
            for key,value in dic.copy().items():
                if value == self.constants_zero[key]:
                    dic.pop(key)
                    
        if include_vdep_consts:
            dic['A_v'] = self.A_v
            dic['B_v'] = self.B_v
            dic['D_v'] = self.D_v
            
        return dic
    
    def update(self,const,exclude_default=True,use_consts=[]):
        """Update the constants defined in the current constants instance
        (:class:`ElectronicStateConstants`) with the ones from another one.

        Parameters
        ----------
        const : :class:`ElectronicStateConstants`
            the instance including the constants values for updating the current
            constants instance.
        exclude_default : bool, optional
            Same keyword argument as in :meth:`to_dict`. The default is True.
        use_consts : list, optional
            If it is desired to only update a certain set of constants, one can
            specify these constants in a list of strings, e.g. ['b_F','c'].
            The default is [].
        """
        # convert constants to dictionary dic
        dic = const.to_dict(include_vdep_consts=False,exclude_default=exclude_default)
        
        # if use_consts does not contain any elements, use all constants in dic
        if len(use_consts) == 0:
            use_consts = list(dic.keys())
        
        for key in use_consts:
            self[key] = dic[key] #update constants from dic
            
    def DunhamCoeffs(self):
        """Handling the Dunham coefficients. Under construction.."""
        pass
    
    def __setitem__(self, index, value):
        if not (index in self.const_all):
            raise KeyError('Only the keys specified in <const_all> can be set!')
        self.__dict__[index] = value
        
    def __getitem__(self, index):
        if not (index in self.const_all):
            raise KeyError('Only the values of the keys specified in <const_all> can be called!') 
        return self.__dict__[index]
    
    def __str__(self):
        return self.show(formatting='non-zero').to_string()
    
    def __repr__(self):
        return repr(self.show(formatting='non-zero'))
    
    def _repr_html_(self):
        return self.show(formatting='non-zero')._repr_html_()
    
    @property
    def b(self):
        return self['b_F'] - self['c']/3
    
    @property
    def b_2(self):
        return self['b_F_2'] - self['c_2']/3
    
    @property
    def A_v(self):
        """returns the vibrational-state-dependent spin-orbit constant `A_v`.
        
        :math:`A_v = A_e + \\alpha_A (v + 1/2)`.
        """
        return self.A_e + self.alpha_A*(self.nu+0.5)
    
    @property
    def B_v(self):
        """returns the vibrational-state-dependent rotational constant `B_v`.
        
        :math:`B_v = B_e - \\alpha_e (v + 1/2) + \gamma_e (v + 1/2)^2`.
        """
        return self.B_e - self.alpha_e*(self.nu+0.5) + self.gamma_e*(self.nu+0.5)**2
    @property
    def D_v(self):
        """returns the vibrational-state-dependent rotational constant `D_v`.
        
        :math:`D_v = D_e + \\beta_e (v + 1/2)`.
        """
        return self.D_e + self.beta_e*(self.nu+0.5)
    @property
    def electrovibr_energy(self):
        """returns the sum of the electronic and vibrational energy.
        
        :math:`E = T_e + \\omega_e (v + 1/2) - \\omega_e \\chi_e (v+1/2)^2 + \\omega_e y_e (v+1/2)^3`.
        """
        nu = self.nu
        w1,w2,w3 = self.w_e, self.__dict__['w_e x_e'], self.__dict__['w_e y_e']
        return self.T_e + w1*(nu+0.5) - w2*(nu+0.5)**2 + w3*(nu+0.5)**3
    
#%%        
class ElectronicState:    
    def __init__(self,label,Smultipl,L,Hcase='a',nu=0,const={},Gamma=None):
        """This class represents an electronic ground or excited state manifold
        which are part of the molecular level structure.
        After an electronic state is created with certain constants of the effective
        Hamiltonian all the single hyperfine states can be added in order to
        calculate the eigenstates and eigenenergies (see :meth:`calc_eigenstates`
        and :meth:`get_eigenstates`).
        
        Parameters
        ----------
        label : str
            label of the electronic state: the first character of this string
            has to be specified as 'X' for a ground state or as 'A', 'B', 'C',
            ... for an excited state.
        Smultipl : int
            spin mulitplicity, i.e. :math:`2S+1`.
        L : int
            orbital angular momentum which defines the type of the electronic
            state as well as the absolute value of the quantum number Lambda.
            Can either be provided as integer :math:`0,1,2,3,...` or as the
            respective Greek symbol :math:`\Sigma,\Pi,\Delta,\Phi,...`.
        Hcase : str, optional
            Hund's case describing the states within the electronic
            state manifold.
            Possible values: 'a' for pure Hund's case a, 'a_p' for parity
            conserved case a, and 'b' for case b. The default is 'a'.
        nu : int, optional
            vibrational quantum number for the vibrational levels.
            The default is 0.
        const : dict or :class:`ElectronicStateConstants`, optional
            dictionary of all constants in wave numbers (1/cm) required for the
            effective Hamiltonian or directly an instance of the class
            :class:`ElectronicStateConstants` (see for further documentation).
            During initialization of the class :class:`ElectronicState`
            an attribute ``const`` as an instance of :class:`ElectronicStateConstants`
            is created. The default is {}.
        Gamma : float, optional
            if the electronic state has the function of an excited state, the
            natural linewidth :math:`\Gamma` must be given for generating a spectrum.
            The natural linewidth Gamma must be given in MHz as an non-angular
            frequency (i.e. Gamma = 1/(2*pi*lifetime)*1e-6. The default is None.
        """
        #determine from label X,A,B,.. if state is electronic ground/ excited state
        if label[0] == 'X':                       
            self.grex = 'ground state'
        elif 'ABCDEFGHIJKLMN'.find(label[0]) >= 0:
            self.grex = 'excited state'
        else:
            raise ValueError('Please provide X,A,B,C,D,.. as first character of `label` for the electronic ground or excited states')
        self.label      = label
        # spin multiplicity and spin
        self.Smultipl   = int(Smultipl)
        self.S          = (self.Smultipl - 1)/2
        # spin-orbital quantum number. Either specified as integer or as Greek name.
        if isinstance(L,(float,int)):
            self.L  = int(L)
        else:
            self.L  = {'Sigma':0, 'Pi':1, 'Delta':2, 'Phi':3, 'Gamma':4}[L]
        # Hund's case
        if not Hcase in ['a','a_p','b','b_betaS']:
            raise Exception('Provided Hunds case {} is not valid!'.format(Hcase))
        self.Hcase      = Hcase
        #constants
        if type(const) == ElectronicStateConstants:
            self.const = const
            self.const.nu = nu
        else:
            self.const = ElectronicStateConstants(const=const,nu=nu)
        
        if Gamma:
            self.Gamma  = Gamma/cm2MHz #in MHz (without 2pi) and then to cm^-1
        else:
            self.Gamma  = None
        
        # vibrational level
        self.__nu         = nu #have to be called after init of self.const
        #self.parity or symmetry for Sigma states --> + or -
        if self.S > 0:  self.shell = 'open'
        else:           self.shell = 'closed'
        #: list for storing the pure states which can be added after class initialization
        self.states = []
        self.states_Hcase = []
        # boolean variable determining if eigenstates are already calculated
        self.eigenst_already_calc = False
        
    def get_energy_casea(self,J,Omega,p):
        """calculate the energy of the electronic state as Hund's case (a).
        The energy is evaluated with an approximate analytic expression
        and returned in units of wave numbers (1/cm).

        Parameters
        ----------
        J : float
            total angular momentum quantum number without nuclear spin.
        Omega : float
            absolute value of the quantum number
            :math:`\Omega = \Lambda + \Sigma`.
        p : int
            parity of the excited state. Either +1 or -1.

        Returns
        -------
        E : float
            energy of the state in wave numbers (1/cm).
        """
        cs = self.const
        nu  = self.nu
        A_v, B_v, D_v = cs.A_v, cs.B_v, cs.D_v
        if Omega > self.L:
            pm      = +1
            Ldoubl  = 0#cs['q'] * (J+0.5)**2
            warnings.warn("Lambda doubling not implemented for Omega > Lambda \
                          and is set to zero for now")
        else:
            pm      = -1
            Ldoubl  = (cs['p']+2*cs['q']) * (J+0.5)
        
        E = cs.electrovibr_energy + pm*A_v*self.L*self.S \
            + (B_v *(J*(J+1) + self.S*(self.S+1) - Omega**2 - self.S**2) - D_v *(J*(J+1))**2) \
            + p*phs(J+0.5) * Ldoubl/2
        return E
    
    def get_energy_caseb(self,N,sr):
        """calculate the energy of the electronic state as Hund's case (b).
        The energy is evaluated with an approximate analytic expression
        and returned in units of wave numbers (1/cm).
        
        Parameters
        ----------
        N : int
            rotational quantum number N.
        sr : int
            Can be either +1 or -1 for the two energy states which are shifted
            up or down in energy respectively due to the spin-rotation interaction.

        Returns
        -------
        E : float
            energy of the state in wave numbers (1/cm).
        """
        cs = self.const
        nu  = self.nu
        B_v, D_v = cs.B_v, cs.D_v
        if sr == +1:   sr = 0.5*cs['gamma']*N
        elif sr == -1: sr = 0.5*cs['gamma']*(-1*(N+1))
        else: raise ValueError('variable <sr> can only take the values +1 or -1')
        
        E = cs.electrovibr_energy \
            + B_v *N*(N+1) - D_v *(N*(N+1))**2 + sr
        return E
    
    def build_states(self,Fmax,Fmin=None):
        """
        builds all the states within an electronic state manifold in the range
        from Fmin to Fmax in units of the total angular momentum quantum number.
        These states are stored in the variable `states` in the instance of this
        class :class:`ElectronicState`. Every time this method is called potentially
        already included states are deleted.

        Parameters
        ----------
        Fmax : float
            upper limit of the total angular momentum quantum number to which 
            all states are added into this instance of :class:`ElectronicState`.
        Fmin : float, optional
            respective lower limit of the total ang. mom. quantum number.
            By default this number is set to the smallest possible number
            which is either 0 or 0.5. The default is None.
        
        Note
        ----
        If `Fmax` or `Fmin` is not properly specified (e.g. when F can only take
        integer values and Fmax=3.5 is provided), it is adjusted to good values instead.
        """
        #for fermions Fmin should be 1/2 due to the second spin --> only 1/2,3/2,5/2,.. possible
        QNrsum = self.S + self.I1 + self.I2 # Lambda and rotational number are leaved out since they are only integers
        if isint(QNrsum):
            Fmin0 = 0
        else:
            Fmin0 = 0.5
        if (Fmin==None) or (Fmin < Fmin0):
            Fmin = Fmin0
        if (int(2*Fmin+0.1)%2 != int(2*Fmin0+0.1)%2):
            Fmin += 0.5
        self.Fmin,self.Fmax = Fmin, np.arange(Fmin,Fmax+1e-3,1)[-1]
        
        self.states = [] #reset states
        self.eigenst_already_calc = False
        if 'Ew' in self.__dict__:
            del self.Ew
            del self.Ev
        
        #___for hunds case a: first case: one nuclear spin; second case: two nuclear spins
        if (self.I1 > 0) and (self.I2 == 0):
            for F in np.arange(Fmin,Fmax+1e-3,1):
                for Si in np.unique([-self.S,self.S]):
                    for L in np.unique([-self.L,self.L]):
                        Om = L + Si
                        for J in addJ(F, self.I1):
                            if J < (abs(Om)-1e-3): continue
                            if (self.Bfield != 0.0) or self.mF_states:
                                for mF in np.arange(-F,F+1e-3,1):
                                    self.states.append(Hcasea(L=L,Si=Si,Om=Om,J=J,F=F,mF=mF,
                                                              S=self.S,I1=self.I1,I2=self.I2))
                            else:
                                self.states.append(Hcasea(L=L,Si=Si,Om=Om,J=J,F=F,
                                                          S=self.S,I1=self.I1,I2=self.I2))
        elif (self.I1 > 0) and (self.I2 > 0):
            for F in np.arange(Fmin,Fmax+1e-3,1):
                for F1 in addJ(F,self.I2):
                    for Si in np.unique([-self.S,self.S]):
                        for L in np.unique([-self.L,self.L]):
                            Om = L + Si
                            for J in addJ(F1, self.I1):
                                if J < (abs(Om)-1e-3): continue
                                if (self.Bfield != 0.0) or self.mF_states:
                                    for mF in np.arange(-F,F+1e-3,1):
                                        self.states.append(Hcasea(L=L,Si=Si,Om=Om,J=J,F1=F1,F=F,mF=mF,
                                                                  S=self.S,I1=self.I1,I2=self.I2))
                                else:
                                    self.states.append(Hcasea(L=L,Si=Si,Om=Om,J=J,F1=F1,F=F,
                                                              S=self.S,I1=self.I1,I2=self.I2))
            
    def calc_eigenstates(self):
        """
        calculates the matrix elements of the various terms of the total
        Hamiltonian and determines the eigenvalues and eigenstates which are
        sorted by energy and stored in the variables ``Ew`` and ``Ev`` in the
        current instace of the class :class:`ElectronicState`.
        The eigenstates can be nicely printed via :meth:`get_eigenstates`.
        
        Warning
        -------
        The total diagonalized Hamiltonian excludes the electronic and vibrational
        constants since the vibrational motion can be decoupled completely from
        the smaller interactions like rotation, hyperfine, spin-orbit,... .
        So, the electronic and vibrational part of the molecular eigenenergies
        are not included in the obtained eigenenergies of this function ``Ew``
        but they can simply be added as an energy offset
        (what is done in the method :meth:`Molecule.calc_branratios`).
        """
        if self.verbose: print('Calc Hamiltonian for {} electronic state'.format(self.label),end=' ')
        H = np.zeros((self.N,self.N))
        const = self.const.to_dict()
        for i,st_i in enumerate(self.states):
            for j in range(i,len(self.states)):
                st_j = self.states[j]
                H[i,j] = H_tot(st_i,st_j,const)
                if self.Bfield != 0.0:
                    H[i,j] += H_Zeeman(st_i,st_j,const,Bfield=self.Bfield)
                #next line can be commented out since H is symmetric and therefore
                #only the upper/lower triangular part of the matrix has to be 
                #used to the calculation of the eigenstates & eigenvalues.
                H[j,i] = H[i,j]
        self.Ham = H #only temporal variable

        if self.verbose: print('..diagonalize it'.format(self.label))     
        if 'mF' in self.states[0].__dict__:
            # store indices of pure states in dictionary ordered by the mF number
            mF_indices = {key : [] for key in np.arange(-self.Fmax,self.Fmax+0.1,1)}
            for i,st in enumerate(self.states):
                mF_indices[st.mF].append(i)
            #test if all states are included somewhere in the dictionary
            count = 0
            for key,value in mF_indices.items():
                count +=len(value)
            if count != self.N: print('WARNING: Not all mF states are included in the dictionary')
            
            #diagonalize only the mF block matrices
            Ew, Ev = np.zeros(self.N),np.zeros((self.N,self.N))
            for mF,indices in mF_indices.items():
                Ew[indices], Ev[np.ix_(indices,indices)] = np.linalg.eigh(H[np.ix_(indices,indices)])
        else:
            try:
                Ew, Ev = np.linalg.eigh(H) #do not use np.linalg.eigh since it yields wrong eigenstates! ??
            except np.linalg.LinAlgError as error:
                print('eigenstate/value calculation did not converge for the first time!!!')
                Ew, Ev = np.linalg.eig(H)
        # sort eigenvalues and eigenstates by energy
        indices = np.argsort(Ew)
        self.Ew = Ew[indices]
        self.Ev = Ev[:,indices]
        
        self.eigenst_already_calc = True
        
    def get_eigenstates(self,rounded=None,onlygoodQuNrs=True,createHTML=False,
                        Hcasebasis=True,mixed_states=False):
        """
        returns the sorted eigenenergies and respective eigenstates determined
        by the method :func:`calc_eigenstates` in a nice format via the datatype
        `pandas.DataFrame` in order to be printed.

        Parameters
        ----------
        rounded : int, optional
            rounded to which the values of the eigenstates are rounded.
            The default is 4.
        onlygoodQuNrs : bool, optional
            specifies if only the good Quantum numbers are included for getting
            a better overview of the printed DataFrame. The default is True.
        createHTML : bool or str, optional
            if True a Html file `eigenstates.html` with the DataFrame is generated
            for a better view of the eigenstates. If a string is given, the file
            will be saved as the respective filename. The default is False.
        Hcasebasis : bool, optional
            If the basis of the eigenenergies is given in pure Hund's case a
            states (False) or in the specified Hund's case basis. The default is True.

        Returns
        -------
        pandas.DataFrame
            the rounded DataFrame comprising the eigenvalues and eigenstates to
            be nicely printed
        """
        if not self.eigenst_already_calc: self.calc_eigenstates()
        
        col_arr = [np.arange(len(self.Ew)),self.Ew]
        if Hcasebasis and (self.Hcase != 'a'):
            DF1 = pd.DataFrame(np.matmul(self.calc_basis_change(),self.Ev), 
                               index=pd.MultiIndex.from_frame(
                                   self.get_states_as_DF(onlygoodQuNrs=onlygoodQuNrs,Hcasebasis=True)),
                               columns=pd.MultiIndex.from_arrays(col_arr,names=('eigenvector i','eigenvalue')))
        else:
            DF1 = pd.DataFrame(self.Ev, 
                               index=pd.MultiIndex.from_frame(
                                   self.get_states_as_DF(onlygoodQuNrs=onlygoodQuNrs)),
                               columns=pd.MultiIndex.from_arrays(col_arr,names=('eigenvector i','eigenvalue')))
        
        if Hcasebasis and (DF1**2).max(axis=0).min() < 0.75:
            warnings.warn("Hcasebasis doesn't seem to be a good choice as the eigenstates are not diagonal (<75%).")
        
        if not mixed_states:
            if not Hcasebasis:
                DF1 = pd.DataFrame(self.Ew, columns=['eigenvalue'],
                                   index=np.arange(len(DF1.index)))#DF1.index.to_frame().iloc[argmax_arr].index)
            else:
                argmax_arr = []
                for (i,Ew),Ev in DF1.items():
                    for argsort_ind in Ev.abs().argsort()[::-1]:
                        if argsort_ind not in argmax_arr:
                            argmax_arr.append( argsort_ind )
                            break
                
                DF1 = pd.DataFrame(self.Ew, columns=['eigenvalue'],
                                   index=DF1.index.to_frame().iloc[argmax_arr].index)
        else:
            DF1 = DF1.sort_index()
            
        if rounded:
            DF1 = DF1.round(rounded)
        if createHTML:
            #render dataframe as html
            html = DF1.to_html()
            #write html to file
            if isinstance(createHTML,str):
                filename = createHTML
            else:
                filename = "eigenstates"
            text_file = open(filename+".html", "w")
            text_file.write(html)
            text_file.close()
        else:
            return DF1
        
    def get_states_as_DF(self,onlygoodQuNrs=False,Hcasebasis=False):
        """returns the states included in the instance :class:`ElectronicState`
        in a nice format via the datatype `pandas.DataFrame` in order to be printed.
        But at first any states have to be added via :func:`build_states`.

        Parameters
        ----------
        onlygoodQuNrs : bool, optional
            specifies if only the good Quantum numbers are included for getting
            a better overview of the printed DataFrame. The default is False.
        Hcasebasis : bool, optional
            specifies whether the pure states or the states in the respective
            Hund's case basis are shown.

        Returns
        -------
        pandas.DataFrame
            the rounded DataFrame comprising the pure states to be nicely printed
        """
        if Hcasebasis:
            if len(self.states_Hcase) == 0:
                self.calc_basis_change()
            states = self.states_Hcase
        else:
            states = self.states
            
        return pd.concat([st.DF(onlygoodQuNrs) for st in states],
                         ignore_index=True)
    
    def calc_basis_change(self):
        '''Calculates the Hund's case states from the pure case a states and determines
        the transformation matrix from the pure state basis to another Hund's case basis.

        Returns
        -------
        np.ndarray
            Transformation matrix from pure to Hund's case basis.
        '''
        self.states_Hcase = []
        self.HcaseBasis = np.zeros((self.N,self.N))
        for i_p,st_p in enumerate(self.states):
            lincom = st_p.to_Hcase(Hcase=self.Hcase)
            for prefac,st_H in zip(lincom['prefacs'],lincom['states']):
                if not st_H in self.states_Hcase:
                    i_H = len(self.states_Hcase)
                    self.states_Hcase.append(st_H)
                else:
                    i_H = self.states_Hcase.index(st_H)
                self.HcaseBasis[i_H,i_p] = prefac
                
        return self.HcaseBasis
    
    def get_gfactors(self, Bmax=1e-4):
        """calculates the mixed g-factors for every hyperfine level eigenstate.
        These g-factors are returned as an array with the same order as the
        eigenstates for zero magnetic field which can be printed via 
        :meth:`get_eigenstates`. The gfactor pd.DataFrame is also stored in the
        attribute ``gfactors``.

        Parameters
        ----------
        Bmax : float, optional
            maximum magnetic field strength in T to which the mixed g-factors are
            calculated. This value should be in the small region where the Zeeman
            shifts possess only linear behavior. The default is 1e-4.

        Returns
        -------
        pandas.DataFrame
            array containing the mixed g-factors ordered by energy of the eigenstates.
        """
        mu_B = physical_constants['Bohr magneton'][0]
        
        oldBfield       = self.Bfield
        mF_states_old   = self.mF_states
        
        # calculate unperturbed eigenvalues without Bfield:
        self.mF_states  = False
        self.Bfield     = 0.0
        self.build_states(self.Fmax, self.Fmin)
        # DataFrames:
        Ew              = self.get_eigenstates()    # eigenvalues without mF states
        gfactors        = Ew.copy()*0.0             # gfactors without mF states 
        MI              = Ew.index.to_frame()       # multiindex without mF states
        
        # calculate new eigenvalues with non-zero Bfield
        self.Bfield     = Bmax
        self.build_states(self.Fmax, self.Fmin)
        # DataFrame: eigenvalues with mFs
        Ew_B            = self.get_eigenstates()
        
        for i, (tmp, row) in enumerate(MI.iterrows()):
            # subgroup DataFrame with all mF state eigenvalues beloning to certain F state.
            Ew_mF   = Ew_B.xs(tuple(row.values), level=tuple(row.keys()), axis=0)
            # a/b := ( eigenvals(B=Bmax) - eigenvals(B=0) ) / (mF_values * mu_B * Bfield)
            a       = Ew_mF.to_numpy()[:,0] - Ew.iloc[i].mean() 
            b       = Ew_mF.index.to_frame().to_numpy()[:,0] * mu_B * Bmax / (cm2MHz * 1e6 * h)
            gfac_all= np.divide(a, b, out=np.empty(a.shape)*np.nan, where=b!=0) # ignore dividing by 0
            if not np.isnan(gfac_all).all(): # only for F != 0
                gfactors.iloc[i] = np.nanmean(gfac_all)
        
        self.gfactors   = gfactors
        
        # set previous values for Bfield and mF_states and build new states
        self.Bfield     = oldBfield
        self.mF_states  = mF_states_old
        self.build_states(Fmax=self.Fmax, Fmin=self.Fmin)
        
        return self.gfactors

    def plot_Zeeman(self,Bfield):
        """plots the Zeeman-splitted levels versus a magnetic field.
        In the plot the eigenvalues are sorted such that the energy crossings
        between different magnetic hyperfine levels are assigned to the right
        curves using the function :func:`eigensort`.

        Parameters
        ----------
        Bfield : array-type or float
            When `Bfield` is of array-type the eigenenergies are calculated for
            every single value. Otherwise, if `Bfield` is a float, the Zeeman
            Hamiltonian is evaluated for 20 values from 0.0 to `Bfield`.
            This input parameter has to be provided in units of Tesla.
        """
        if not isinstance(Bfield, Iterable):
            Bfield = np.linspace(0,Bfield,20) #create simple array with maximum value Bfield
        if Bfield[0] == 0.0:  # prevent the first Bfield value to be zero
            Bfield[0] = Bfield[1]*1e-4
        
        # don't change states and magnetic field settings in self
        ElSt = deepcopy(self)
        
        ElSt.mF_states  = True
        ElSt.build_states(ElSt.Fmax, ElSt.Fmin)
        
        Ew_B_arr = np.zeros((len(Bfield),ElSt.N))
        for k,B in enumerate(Bfield):
            ElSt.Bfield = B
            ElSt.calc_eigenstates()
            Ew_B_arr[k,:] = ElSt.Ew
        
        # plotting 
        Ew_B_arr = eigensort(Bfield, Ew_B_arr)
        plt.figure()
        for i in range(Ew_B_arr.shape[1]):
            plt.plot(Bfield*1e4, Ew_B_arr[:,i], '-')
            
        plt.xlabel('Magnetic field (G)')
        plt.ylabel('Energy (cm$^{-1}$)')
        
        return plt.gca()
        
    def export_OBE_properties(self, index_filter={}, rounded=None, QuNrs=[], 
                              HFfreq_offset=0, Bmax=1e-4, nested_dict=False,
                              get_QuNr_keyval_pairs_kwargs={}):
        """Export all the important properties connected to a single electronic
        state to a dictionary in a proper format for the OBE simulation code to
        import the properties from the corresponding json file. This method is
        e.g. used in the similar function :meth:`Molecule.export_OBE_properties`
        from the :class:`Molecule.Molecule` class to also save the whole
        json file with all properties.

        Parameters
        ----------
        index_filter : dict, optional
            to filter the states of interest. See `rows` argument of
            :func:`multiindex_filter`. The default is {}.
        rounded : int, optional
            digit to round the frequencies and gfactors. The default is None
            meaning that no rounding is applied.
        QuNrs : list(str), optional
            Which Quantum numbers should be exported into the output dictionary.
            The default is [] meaning that only the necessary unique Quantum numbers
            are used (see :func:`get_unique_multiindex_names`).
        HFfreq_offset : float, optional
            applying an offset to the whole array of hyperfine frequencies (MHz)
            whose lowest eigenvalue is always normalized to 0. The default is 0.
        Bmax : float, optional
            Determines how the gfactors are calculated (see :meth:`get_gfactors`).
            The default is 1e-4.
        nested_dict : bool, optional
            Whether the raw dictionary with the data or properties should be nested
            into other dictionaries required by the json file. See
            :func:`get_QuNr_keyval_pairs`. The default is False.
        get_QuNr_keyval_pairs_kwargs : kwargs, optional
            keyword arguments for :func:`get_QuNr_keyval_pairs`. The default is {}.

        Returns
        -------
        dict
            Dictionary with the properly formatted ElectronicState level properties.
        """
        def DF2list(df): # round and convert DataFrame to list for saving in json dict
            if rounded != None:
                df = df.round(rounded)
            return list(df.to_numpy()[:,0])
        
        old_values = {key:self.__dict__[key] for key in ['mF_states', 'Bfield']}
        self.mF_states = False
        self.Bfield    = 0
        self.build_states(self.Fmax)
        
        Ew      = multiindex_filter(self.get_eigenstates(), rows=index_filter, drop_level=False)
        Ew      = (Ew-Ew.min())*cm2MHz + HFfreq_offset
        gfac    = multiindex_filter(self.get_gfactors(Bmax), rows=index_filter, drop_level=False)
        if not QuNrs:
            QuNrs = get_unique_multiindex_names(Ew.index)
            
        OBE_props   = dict(gfac    = DF2list(gfac),
                           HFfreq  = DF2list(Ew),
                           QuNrs   = get_QuNrvals_from_multiindex(Ew.index, QuNrs))
        
        if nested_dict:
            key_list    = get_QuNr_keyval_pairs(self, Ew.index, **get_QuNr_keyval_pairs_kwargs)
            OBE_props_  = OBE_props.copy()
            OBE_props   = NestedDict()
            OBE_props[key_list] = OBE_props_
            if self.grex == 'excited state':
                OBE_props[key_list[0]].update(dict(Gamma=self.Gamma*cm2MHz))
            
        self.__dict__.update(old_values)
        self.build_states(self.Fmax)
        
        return OBE_props
        
    def __str__(self):
        """prints all general information of the ElectronicState"""
        Lnames = ['Sigma','Pi','Delta','Phi','Gamma']
        linewidth = ''
        if self.Gamma:
            linewidth += '\n  - linewidth: 2 pi * {:.2f} MHz'.format(self.Gamma*cm2MHz)
        return '{:13s} {}(^{}){}:\n  - Hunds case {}\n  - {} shell electronic state'.format(
            self.grex,self.label,self.Smultipl,Lnames[self.L],self.Hcase,self.shell) \
            + linewidth + '\n  --> includes {} states'.format(self.N)

    @property
    def nu(self):
        """vibrational quantum number for the vibrational levels.
        Can be called and changed.
        """
        return self.__nu # maybe check if self.const.nu is different?
    
    @nu.setter
    def nu(self,val):
        if (val == self.const.nu) & (val == self.__nu): return
        if not isint(val):
            raise ValueError('given value {} is no integer!'.format(val))
        self.const.nu = val
        self.__nu = val
        self.eigenst_already_calc = False
        
    @property
    def N(self):
        """returns the number of states in the current instance of :class:`ElectronicState`.

        Returns
        -------
        int
            number of states 
        """
        return len(self.states)
    
#%%
class QuState:
    goodQuNrs = []
    description = 'State without certain Hunds case'
    
    def __init__(self,**kwargs):
        """Instance of the class represents a molecular state as Hund's case a.

        Parameters
        ----------
        **kwargs : float
            quantum numbers of the Hund's case a.
        """
        # self.L,self.Si,self.Om,self.J,self.F,self.I1 = L,Si,Om,J,F,I1
        self.__dict__.update(kwargs)
        self.QuNrs = list(kwargs.keys()) # convert to list since otherwise an error arises with pickle
        # check if all quantum numbers except L and Si are positive!!?
        self.linearcombi = {}
    
    def DF(self,onlygoodQuNrs=False):
        """return a DataFrame as nice representation of the state with all quantum numbers.

        Parameters
        ----------
        onlygoodQuNrs : bool, optional
            if all quantum numbers or only the good ones are shown. The default is False.

        Returns
        -------
        pandas.DataFrame
            DataFrame showing the quantum numbers.
        """
        if onlygoodQuNrs:
            QuNrs = self.goodQuNrs[:]
            if self.I2 != 0:
                QuNrs += ['F1']
            QuNrs += ['F']
            if 'mF' in self.QuNrs: 
                QuNrs += ['mF']
        else:
            QuNrs = self.QuNrs
        return pd.DataFrame([[self.__dict__[Nr] for Nr in QuNrs]],columns=QuNrs) 

    def __str__(self):
        return self.DF().to_string() + '\n' + self.description

    def __eq__(self, other):
        if len(self.QuNrs) != len(other.QuNrs):
            return False
        else:
            for QuNr in self.QuNrs:
                if self.__dict__[QuNr] != other.__dict__[QuNr]:
                    return False
        return True
    
    def QuNrs_default(self): 
        QuNrs_def = {}
        for QuNr in ['F','F1','mF','S','I1','I2']:
            if QuNr in self.QuNrs:
                QuNrs_def[QuNr] = self.__dict__[QuNr]
        return QuNrs_def
        
class Hcasea(QuState):
    #: good QuNrs for a pure Hund's case a
    goodQuNrs = ['L','Si','Om','J'] #different for Fermions and bosons? #difference between L and La (Lambda)?
    description = 'Hunds case a (pure state)'
    
    def to_Hcase(self,Hcase='b',printing=False):
        """transforms the pure Hund's case a state into another Hund's case basis.        

        Parameters
        ----------
        Hcase : str, optional
            Hund's case basis for transformation. The default is 'b'.
        printing : bool, optional
            if the linear combination of the states of the new basis is printed.
            The default is False.

        Returns
        -------
        dict
            linear combination of the states of the new Hund's case basis as a
            dictionary with the keys `prefacs` and `states`.
        """
        QuNrs_def = self.QuNrs_default()
        
        if not Hcase in self.linearcombi:
            states, prefacs  = [], []
            if Hcase == 'a':
                states.append(self)
                prefacs.append(+1)
            elif Hcase == 'a_p':
                for P,prefac in zip([+1,-1], np.array([+1,np.sign(self.L)])/np.sqrt(2)):
                    if prefac == 0: continue
                    if np.sign(self.L) < 0: prefac *= (-1)**(self.J-self.S) # maybe additional -1 factor for reflection symmetry +-
                    states.append(Hcasea_p(L=abs(self.L),P=P,Om=abs(self.Om),J=self.J,**QuNrs_def))
                    prefacs.append(prefac)
            elif Hcase == 'b' or Hcase == 'b_betaS':
                for N in addJ(self.S,self.J):
                    prefac = np.sqrt(2*N+1)*phs(self.J+self.Om)*w3j(self.S,N,self.J,self.Si,self.L,-self.Om)
                    if prefac == 0: continue
                    st = Hcaseb(L=self.L,N=N,J=self.J,**QuNrs_def)
                    if Hcase == 'b_betaS':
                        lincom_ = st.to_Hcase(Hcase='b_betaS')
                        for prefac_,st_ in zip(lincom_['prefacs'],lincom_['states']):
                            if st_ in states:
                                prefacs[states.index(st_)] += prefac_*prefac
                            else:
                                states.append(st_)
                                prefacs.append(prefac_*prefac)
                    else:
                        states.append(st)
                        prefacs.append(prefac)
            else: # Transformation for the fermions is missing?!
                raise ValueError("Invalid string value '{}' for `Hcase` parameter".format(Hcase))
            self.linearcombi[Hcase] = dict(states=states, prefacs=prefacs)
        
        if printing:
            for prefac, state in zip(self.linearcombi[Hcase]['prefacs'],
                                     self.linearcombi[Hcase]['states']):
                print('{:+.4f} *\n{}'.format(prefac,state,end='\n'))
        
        return self.linearcombi[Hcase]

class Hcasea_p(QuState):
    #: good QuNrs for a Hund's case a (parity conserved)
    goodQuNrs = ['P','Om','J'] #different for Fermions and bosons?
    description = 'Hunds case a (parity conserved)'
    
class Hcaseb(QuState):
    #: good QuNrs for a pure Hund's case b
    goodQuNrs = ['L','N','J'] #different for Fermions and bosons?
    description = 'Hunds case b_betaJ'
    
    def to_Hcase(self,Hcase='b_betaS',printing=False):
        """transforms the pure Hund's case a state into another Hund's case basis.        

        Parameters
        ----------
        Hcase : str, optional
            Hund's case basis for transformation. The default is 'b_betaS'.
        printing : bool, optional
            if the linear combination of the states of the new basis is printed.
            The default is False.

        Returns
        -------
        dict
            linear combination of the states of the new Hund's case basis as a
            dictionary with the keys `prefacs` and `states`.
        """
        QuNrs_def   = self.QuNrs_default()
        if self.I2 != 0:    F = self.F1
        else:               F = self.F
        
        if not Hcase in self.linearcombi:
            states, prefacs  = [], []
            if Hcase == 'b':
                states.append(self)
                prefacs.append(+1)
            elif Hcase == 'b_betaS':
                for G in addJ(self.S,self.I1):
                    prefac = np.sqrt((2*self.J+1)*(2*G+1))*phs(self.N+self.S+F+self.I1)\
                             * w6j(self.N,self.S,self.J,
                                   self.I1,    F,     G)
                    if prefac == 0: continue
                    states.append(Hcaseb_betaS(L=self.L,N=self.N,G=G,**QuNrs_def))
                    prefacs.append(prefac)
            else: # Transformation for the fermions is missing?!
                raise ValueError("Invalid string value '{}' for `Hcase` parameter".format(Hcase))
            self.linearcombi[Hcase] = dict(states=states, prefacs=prefacs)
        
        if printing:
            for prefac, state in zip(self.linearcombi[Hcase]['prefacs'],
                                     self.linearcombi[Hcase]['states']):
                print('{:+.4f} *\n{}'.format(prefac,state,end='\n'))
        
        return self.linearcombi[Hcase]

class Hcaseb_betaS(QuState):
    goodQuNrs = ['L','N','G']
    description = 'Hunds case b_betaS'

#%% Hamiltonians
def H_tot(x,y,const):
    """calculates and returns the matrix element of the total Hamiltonian without
    external fields between two states.

    Parameters
    ----------
    x : :class:`Hcasea`
        first state.
    y : :class:`Hcasea`
        second state.
    const : dict
        dictionary of all constants required for the effective Hamiltonian.
        When this function is called by the method :meth:`ElectronicState.calc_eigenstates`,
        the method :meth:`ElectronicStateConstants.to_dict` of the attribute
        ``const`` within :class:`ElectronicState` is used to
        create a proper dictionary.
    """
    # x: lower state, y: upper state
    # prevent mixing of different F values
    if kd(x.F,y.F) == 0: return 0.0
    S   = x.S
    L, Si, Om, J  = x.L, x.Si, x.Om, x.J
    L_,Si_,Om_,J_ = y.L, y.Si, y.Om, y.J
    if x.I2 > 0:
        I1  = x.I1
        I2  = x.I2
        F, F1   = x.F, x.F1
        F_,F1_  = y.F, y.F1
        #==================== H_hfs2 - hyperfine interaction for both nuclear ang. moments.
        sum1 = 0.0
        for q in [-1,0,+1]:
            sum1 += phs(J-Om)*w3j(J,1,J_,-Om,q,Om_)*(
                const['a_2']*kd(Si,Si_)*kd(Om,Om_)*L_
                + const['b_F_2']*phs(S-Si)*cb(S)*w3j(S,1,S,-Si,q,Si_)
                + const['c_2']*np.sqrt(30)/3*phs(q+S-Si)*cb(S)*w3j(S,1,S,-Si,q,Si_)*w3j(1,2,1,-q,0,q)
                )
        # kd(L,L_): Delta Lambda may not be strictly true
        term1 = kd(L,L_)*sum1
        
        sum2 = 0.0
        for q in [-1,+1]:
            sum2 += kd(L, L_-2*q)*phs(J-Om+q+S-Si)*cb(S) \
                    *w3j(J,1,J_,-Om,-q,Om_)*w3j(S,1,S,-Si,q,Si_)
        term2 = -const['d_2']*sum2
        
        H_hfs2 = phs(F1_+I2+F)*phs(J+I1+F1_+1)*cb(I2)*sb(F1)*sb(F1_)*sb(J)*sb(J_) \
                *w6j(I2,F1_,F,F1,I2,1)*w6j(J_,F1_,I1,F1,J,1) * (term1 + term2)
        
        # exit if Delta F1 != 0     <-- why can one do that?
        if kd(F1,F1_) == 0: return H_hfs2
        
        #In the following Hailtonians F1 is refered to as F, and I1 as I
        F   = F1 
        F_  = F1_
        I   = I1
        
    else:
        F   = x.F
        F_  = y.F
        I   = x.I1
        H_hfs2 = 0.0
    
    #==================== H_hfs - hyperfine structure 
    sum1 = 0.0
    for q in [-1,0,+1]:
        sum1 += phs(J-Om)*w3j(J,1,J_,-Om,q,Om_)*(
            const['a']*kd(Si,Si_)*kd(Om,Om_)*L_
            + const['b_F']*phs(S-Si)*cb(S)*w3j(S,1,S,-Si,q,Si_)
            + const['c']*np.sqrt(30)/3*phs(q+S-Si)*cb(S)*w3j(S,1,S,-Si,q,Si_)*w3j(1,2,1,-q,0,q)
            )
    term1 = kd(L,L_)*phs(J_+I+F)*cb(I)*sb(J)*sb(J_)*w6j(F,J_,I,1,I,J) * sum1

    sum2 = 0.0
    for q in [-1,+1]:
        sum2 += kd(L, L_-2*q)*phs(J_+I+F)*phs(J-Om+q+S-Si)*cb(I)*cb(S)*sb(J)*sb(J_) \
                *w6j(F,J_,I,1,I,J)*w3j(J,1,J_,-Om,-q,Om_)*w3j(S,1,S,-Si,q,Si_)
    term2 = -const['d']*sum2
    
    term3 = const['c_I']/2* (F*(F+1)-I*(I+1)-J*(J+1))* \
            kd(Si,Si_)*kd(L,L_)*kd(J,J_) #only diagonal terms considered here
    H_hfs = term1 + term2 + term3
    
    #==================== H_rot - rotational term 
    H_rot = 0.0
    if (const['B_v'] != 0.0) and (kd(J,J_) != 0.0):
        sum1 = 0.0
        for q in [-1,+1]:
            sum1 += phs(J_-Om+S-Si)*cb(J)*cb(S)*w3j(J,1,J,-Om,q,Om_)*w3j(S,1,S,-Si,q,Si_)
        H_rot = const['B_v']*kd(J,J_)*(
            -2*sum1 + kd(Si,Si_)*kd(Om,Om_)*(J*(J+1) + S*(S+1) - Om_**2 - Si_**2) )        
    
    #==================== H_sr - spin-rotation coupling
    H_sr = 0.0
    if (const['gamma'] != 0.0) and (kd(J,J_) != 0.0):
        sum1 = 0.0
        for q in [-1,+1]: #same as in H_rot
            sum1 += phs(J_-Om+S-Si)*cb(J)*cb(S)*w3j(J,1,J,-Om,q,Om_)*w3j(S,1,S,-Si,q,Si_)
        H_sr = const['gamma']*kd(J,J_)*(
            sum1 + kd(Si,Si_)*kd(Om,Om_)*(Si**2 - S*(S+1))  )
    # no first order contribution for excited states ????
    if not ((L==0) and (L_==0)): H_sr = 0.0
    
    #==================== H_so - spin-orbit coupling
    H_so = 0.0
    if (kd(J,J_) != 0.0):
        H_so = kd(L,L_)*kd(Om,Om_)*kd(J,J_)*kd(Si,Si_)*(
            const['A_v']*L*Si
            + const['A_D']*L*Si* (J*(J+1) - Om**2 + S*(S+1) - Si**2)   )
    
    #==================== H_LD - Lambda-doubling
    H_LD = 0.0
    if ((const['o']!=0.0) or (const['p']!=0.0) or (const['q']!=0.0)) and (kd(J,J_) != 0.0):
        sum1 = 0.0
        for q in [-1,+1]:
            sum11 = 0.0
            for Si__ in np.unique([-S,S]): #or is np.arange(-S,S+1e-3,1) better in general?
                sum11 += phs(S-Si+S-Si__)*cb(S)*w3j(S,1,S,-Si,q,Si__)*w3j(S,1,S,-Si__,q,Si_)
            sum12 = phs(J_-Om+S-Si)*cb(S)*cb(J_)*w3j(J_,1,J_,-Om,-q,Om_)*w3j(S,1,S,-Si,q,Si_)
            sum13 = 0.0
            Om_max = abs(Si)+abs(L)
            for Om__ in np.arange(-Om_max,Om_max+1e-3,1):
                sum13 += phs(J_-Om+J_-Om__)*cb(J_)*w3j(J,1,J,-Om,-q,Om__)*w3j(J,1,J,-Om__,-q,Om_)
                
            sum1 += kd(L,L_-2*q)*(  (const['o']+const['p']+const['q']) * kd(Om,Om_)*sum11 
                                  + (const['p']+2*const['q']) * sum12
                                  +  const['q'] * kd(Si,Si_)*sum13   )
        H_LD = kd(J,J_) * sum1        
    
    #==================== H_eq0Q - electric quadrupol
    H_eq0Q = 0.0
    par1 = w3j(I,2,I,-I,0,I)
    if (par1 != 0.0) and (const['eq0Q'] != 0.0):
        H_eq0Q = const['eq0Q']/4*kd(L,L_)*kd(Om,Om_)*phs(J_+I+F)*phs(J-Om_)*sb(J)*sb(J_) \
                *1/par1*w6j(F,J_,I,2,I,J)*w3j(J,2,J_,-Om,0,Om)
    
    
    return H_hfs + H_rot + H_sr + H_so + H_LD + H_eq0Q + H_hfs2

def H_Zeeman(x,y,const,Bfield):
    """calculates and returns the matrix element of the Zeeman interaction
    between two states.

    Parameters
    ----------
    x : :class:`Hcasea`
        first state.
    y : :class:`Hcasea`
        second state.
    const : dict
        dictionary of all constants required for the effective Hamiltonian.
        When this function is called by the method :meth:`ElectronicState.calc_eigenstates`,
        the method :meth:`ElectronicStateConstants.to_dict` of the attribute
        ``const`` within :class:`ElectronicState` is used to
        create a proper dictionary.
    Bfield : float
        magnetic field strength in T.
    """
    # x: lower state, y: upper state
    # prevent mixing of different mF values
    if kd(x.mF,y.mF) == 0: return 0.0
    S   = x.S
    L, Si, Om, J  = x.L, x.Si, x.Om, x.J
    L_,Si_,Om_,J_ = y.L, y.Si, y.Om, y.J
    unit = 0.4668644778272809#mu_B/h*1e-6/cm2MHz #Bfield*mu_B=E, E/h=f, f in MHz -> cm^-1
    if x.I2 > 0:
        F, mF, F1   = x.F,x.mF,x.F1
        F_,mF_,F1_  = y.F,y.mF,y.F1
        I1,I2       = x.I1,x.I2
        
        sum1 = 0.0
        for q in [-1,0,+1]:
            sum1 += w3j(J,1,J_,-Om,q,Om_)*( const["g'_L"]*L*kd(Si,Si_) #-const['g_l']*kd(Si,Si_)*Si #see Brown&Carrington, but not used in Fortran code
                + (const['g_S']+const['g_l'])*phs(S-Si)*cb(S)*w3j(S,1,S,-Si,q,Si_) )
        H_z1 = Bfield*kd(L,L_)*sum1
        
        sum2 = 0.0
        for q in [-1,+1]:
            sum2 += kd(L,L_-2*q)*w3j(S,1,S,-Si,q,Si_)*w3j(J,1,J_,-Om,-q,Om_)
        H_z2 = -Bfield*const["g'_l"]* phs(S-Si)*cb(S)*sum2
        
        #common factor
        common_fac = phs(F-mF+J-Om)*sb(J)*sb(J_)*w3j(F,1,F_,-mF,0,mF)\
                    *sb(F1)*sb(F1_)*w6j(J,F1,I1,F1_,J_,1)*phs(F1_+J+I1+1)\
                    *sb(F)*sb(F_)*w6j(F1,F,I2,F_,F1_,1)*phs(F_+F1+I2+1)
        return (H_z1 + H_z2)*common_fac*unit

    else:
        F, mF   = x.F,x.mF
        F_,mF_  = y.F,y.mF
        I       = x.I1
        
        sum1 = 0.0
        for q in [-1,0,+1]:
            sum1 += w3j(J,1,J_,-Om,q,Om_)*( const["g'_L"]*L*kd(Si,Si_)
                + (const['g_S']+const['g_l'])*phs(S-Si)*cb(S)*w3j(S,1,S,-Si,q,Si_) )
        H_z1 = Bfield*kd(L,L_)*sum1
        
        sum2 = 0.0
        for q in [-1,+1]:
            sum2 += kd(L,L_-2*q)*w3j(S,1,S,-Si,q,Si_)*w3j(J,1,J_,-Om,-q,Om_)
        H_z2 = -Bfield*const["g'_l"]*phs(S-Si)*cb(S)*sum2
        
        #common factor
        common_fac = phs(F-mF+J-Om)*sb(J)*sb(J_)*w3j(F,1,F_,-mF,0,mF)\
                    *sb(F)*sb(F_)*w6j(J,F,I,F_,J_,1)*phs(F_+J+I+1)
        
        return (H_z1 + H_z2)*common_fac*unit

def H_d(x,y):
    """calculates and returns the matrix element of the electric dipole
    operator between two states.

    Parameters
    ----------
    x : :class:`Hcasea`
        first state.
    y : :class:`Hcasea`
        second state.
    """
    # x: lower state, y: upper state
    S   = x.S
    L, Si, Om, J ,F  = x.L, x.Si, x.Om, x.J, x.F
    L_,Si_,Om_,J_,F_ = y.L, y.Si, y.Om, y.J, y.F
    if kd(Si,Si_) == 0.0: return 0.0
    
    if x.I2 > 0: # for two nuclear spins 
        I1,I2   = x.I1, x.I2
        F1,F1_  = x.F1, y.F1
        
        sum1 = 0.0
        for q in [-1,0,+1]:
            sum1 += w3j(J_,1,J,-Om_,q,Om)       
        H = kd(Si,Si_)*phs(F+F1_+I2+1)*sb(F)*sb(F_)*w6j(F1,F,I2,F_,F1_,1) \
            *phs(F1+J_+I1+1)*sb(F1)*sb(F1_)*w6j(J,F1,I1,F1_,J_,1) \
            *phs(J_-Om_)*sb(J)*sb(J_) * sum1
        
    else: # for one nuclear spin
        I   = x.I1
        
        sum1 = 0.0
        for q in [-1,0,+1]: #here also add 0?
            sum1 += w3j(J_,1,J,-Om_,q,Om)
        H = kd(Si,Si_)*phs(J_+I+F+1)*sb(F)*sb(F_)*w6j(J_,F_,I,F,J,1)*phs(J_-Om_)*sb(J)*sb(J_)*sum1
    if 'mF' in x.QuNrs:
        mF,mF_ = x.mF, y.mF
        H *= phs(F_-mF_)*w3j(F_,1,F,-mF_,mF_-mF,mF)
    return H

#%%% small functions
@jit(nopython=True,parallel=False)
def addJ(J1,J2):
    """adding two angular momenta. Returns an array of all possible total
    angular momenta."""
    ishalfint(J1,raise_err=True)
    ishalfint(J2,raise_err=True)
    return np.arange(np.abs(J1-J2),np.abs(J1+J2)+1e-3,1)
@jit(nopython=True,parallel=False)
def cb(x):
    """curly brackets expression in the Hamiltonian"""
    # curly brackets
    ishalfint(x,raise_err=True)
    return np.sqrt(x*(x+1)*(2*x+1))
@jit(nopython=True,parallel=False)
def sb(x):
    """square brackets expression in the Hamiltonian"""
    ishalfint(x,raise_err=True)
    return np.sqrt(2*x+1)
@jit(nopython=True,parallel=False)
def phs(x):
    """phase expression '(-1)^(x) in the Hamiltonian"""
    if not isint(x):
        raise ValueError('no real phase (-1)**(x) for the value x')#'={}'.format(x))
    return (-1)**int(x+0.1)
@jit(nopython=True,parallel=False)
def kd(x,y):
    """delta kronecker"""
    ishalfint(x,raise_err=True)
    ishalfint(y,raise_err=True)
    if np.abs(x-y) < 1e-13: return 1
    else: return 0
@jit(nopython=True,parallel=False)
def isint(x,raise_err=False):
    """test if x is an integer value and raise an error if it's intended"""
    if abs(np.around(x) - x) > 1e-13:
        if raise_err: raise ValueError('No integer is provided!!')
        return False
    else: return True
@jit(nopython=True,parallel=False)
def ishalfint(x,raise_err=False):
    """test if x is a half-integer value and raise an error if it's intended"""
    if abs(np.around(2*x) - 2*x) > 1e-13:
        if raise_err: raise ValueError('No half-integer is provided!!')
        return False
    else:
        return True

def eigensort(x,y_arr):
    """sorts an eigenvalue matrix, e.g. eigenvalues as a function of a
    varying magnetic or eletric field. This is especially useful since
    crossing curves of eigenvalues are rearanged in the right order.

    Parameters
    ----------
    x : 1D numpy array and length N
        x array, e.g. for the varying magnetic of electric field.
    y_arr : 2D numpy array of shape(N,M)
        Array containing all eigenvalues for each value of x.

    Returns
    -------
    y_arr : 2D numpy array of shape(N,M)
        ordered array.
    """
    # y_arr should have the shape(len(x),N) with N > 2 arbitrary length
    # first sort all eigenvalues
    y_arr = np.sort(y_arr,axis=1)
    for i in range(2,y_arr.shape[0]):
        slope = (y_arr[i-1,:]-y_arr[i-2,:])/(x[i-1]-x[i-2])
        ind_arr = [] # indices array
        for j in range(y_arr.shape[1]):
            inds = np.argsort(np.abs(
                    y_arr[i-1,j]+slope[j]*(x[i]-x[i-1]) - y_arr[i,:] ))
            for ind in inds:
                if ind not in ind_arr:
                    ind_arr.append(ind)
                    break
        ind_arr = np.array(ind_arr)
        y_arr[i,:] = y_arr[i,ind_arr]
    return y_arr

def multiindex_filter(DF,rows=dict(),cols=dict(), drop_level=True):
    """Filter a DataFrame object only for few specific multiindex values.

    Parameters
    ----------
    DF : pd.DataFrame
        DataFrame which is going to be filtered.
    rows : dict, optional
        rows to be retained, i.e. dict(N=1). The default is dict().
    cols : dict, optional
        columns to be retained, i.e. dict(P=1,Om=0.5,J=0.5). The default is dict().
    drop_level : bool, optional
        If redundant index levels should be dropped. The default is True.

    Returns
    -------
    DF : pd.DataFrame
        filtered DataFrame.
    """
    if rows:
        DF = DF.xs(tuple(rows.values()), level=tuple(rows.keys()), axis=0, drop_level=drop_level)
    if cols:
        DF = DF.xs(tuple(cols.values()), level=tuple(cols.keys()), axis=1, drop_level=drop_level)
    return DF

def get_QuNrvals_from_multiindex(multiindex, QuNrs):
    """Extract lists with the values certain Quantum numbers from a
    pandas.MultiIndex object.

    Parameters
    ----------
    multiindex : pandas.MultiIndex
        MultiIndex with one or multiple columns corresponding to Quantum numbers.
    QuNrs : list(str)
        names of the Quantum number for which the values should be extracted.

    Returns
    -------
    d : dict
        dictionary with Quantum number names as keys and the Quantum number values
        as lists.
    """
    d = dict()
    for QuNr in QuNrs[:]:
        QuNrs_arr = list(multiindex.to_frame().get(QuNr).to_numpy())
        if np.all(list(map(isint, QuNrs_arr))):
            QuNrs_arr = list(map(int, QuNrs_arr))
        d[QuNr] = QuNrs_arr
    return d

def get_QuNr_keyval_pairs(ElState, multiindex, QuNrs_names=[], include_v=True):
    """Calculated list of pairs of the name and value of certain Quantum numbers
    included in a MultiIndex object.

    Parameters
    ----------
    ElState : :class:`ElectronicState`
        ElectronicState for which the vibrational and ground or excited data is
        extracted.
    multiindex : pandas.MultiIndex
        MultiIndex object which includes multiple rows of Quantum number values
        for multiple Quantum number names columns.
    QuNrs_names : list(str), optional
        all Quantum number names to be ectraced. The default is [].
    include_v : bool, optional
        Whether to include the vibrational Quantum numbers. The default is True.
    
    Example
    -------
    ::
        
        >>> get_QuNr_keyval_pairs(Mol.A, Mol.A.get_eigenstates().iloc[:10].index, QuNrs_names=['Om'])
        Out: ['exs=A', 'v=0', 'Om=0.5']
        
    Returns
    -------
    list(str)
        list of Quantum number key and value pairs, e.g. ['gs=X', 'v=0', 'N=1'].
    """
    if isinstance(ElState, list):
        if len(ElState) != 2:
            raise Exception(f"<ElState> list must be of len 2, not {len(ElState)}")
        if QuNrs_names == []:
            QuNrs_names = [[],[]]
        pairs_list = [get_QuNr_keyval_pairs(ElState[i], multiindex[i], QuNrs_names[i])
                      for i in range(2)]
        return [f"{str_g} <- {str_e}" for str_g, str_e in zip(*pairs_list)]
    
    gsexs   = {'ground state':'gs', 'excited state':'exs'}[ElState.grex]
    # list of QuNrs label and value pairs
    pairs   = [f"{gsexs}={ElState.label}"]
    if include_v:
        pairs.append(f"v={ElState.nu}")
    # add values to the names/ labels of the Quantum numbers
    for QuNr_name in QuNrs_names[:]:
        values = multiindex.get_level_values(QuNr_name)
        if not np.all(values == values[0]):
            raise Exception(f"not all values {values} are the same for QuNr {QuNr_name}")
        value = int(values[0]) if isint(values[0]) else values[0]
        pairs.append(f"{QuNr_name}={value}")
    return pairs

def get_unique_multiindex_names(multiindex):
    """Calculate list of unique column names from a pandas.MultiIndex object."""
    multiindex      = multiindex.copy()
    nunique         = multiindex.to_frame().nunique()
    cols_to_drop    = nunique[nunique == 1].index
    multiindex      = multiindex.droplevel(list(cols_to_drop.values))
    return list(multiindex.names)

class NestedDict(dict):
    """Nested dictionary class to handle multiple key with the corresponding
    ``__getitem__`` and ``__setitem__`` methods.
    
    Example
    -------
    ::
        
        >>> f1 = NestedDict()
        >>> f1
        {}
        >>> f1[['a','b','c']]
        >>> f1
        {'a': {'b': {'c': {}}}}
        >>> f1[['a2','b2','c2']] =1234
        >>> f1
        {'a': {'b': {'c': {}}}, 'a2': {'b2': {'c2': 1234}}}
        >>> f1[['a','b']].update(dict(c2=1234))
        >>> f1
        {'a': {'b': {'c': {}, 'c2': 1234}}, 'a2': {'b2': {'c2': 1234}}}
    """
    def __getitem__(self, item):
        if isinstance(item, list):
            if len(item) > 1:
                return self[item[0]][item[1:]]
            else:
                item = item[0]
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
        
    def __setitem__(self, key, value):
        # optional processing here
        if isinstance(key, list):
            if len(key) > 1:
                self[key[0]][key[1:]] = value
            else:
                self[key[0]] = value
        else:
            super(NestedDict, self).__setitem__(key, value)
#%%
if __name__ == '__main__':
    # BaF constants for the isotopes 138 and 137 from Steimle paper 2011
    const_gr_138 = {'B_e':0.21594802,'D_e':1.85e-7,'gamma':0.00269930,
                    'b_F':0.002209862,'c':0.000274323} #,'g_l':-0.028}#gl constant??
    const_ex_138 = {'A_e':632.28175,'A_D':3.1e-5,
                    'B_e':0.2117414, 'D_e':2.0e-7,'p':-0.25755-2*(-0.0840),'q':-0.0840,
                    "g'_l":-0.536,"g'_L":0.980,'T_e':11946.31676963}
    
    const_gr_137 = {'B_e':0.21613878,'D_e':1.85e-7,'gamma':0.002702703,
                    'b_F':0.077587, 'c':0.00250173,'eq0Q':-0.00390270*2,
                    'b_F_2':0.002209873,'c_2':0.000274323}
    const_ex_137 = {'B_e':0.211937,'D_e':2e-7,'A_e':632.2802,'A_D':3.1e-5,
                    'p':-0.2581-2*(-0.0840),'q':-0.0840,'d':0.0076,
                    'T_e':11946.3152748}
    Gamma       = 2.8421
    mass138     = 138 + 19
    mass137     = 137 + 19
    #%% bosonic 138BaF
    BaF = Molecule(I1=0.5,naturalabund=0.717,label='138 BaF',mass=mass138)
    BaF.add_electronicstate('X',2,'Sigma', Hcase='b',const=const_gr_138) #for ground state
    BaF.add_electronicstate('A',2,'Pi',Gamma=Gamma, Hcase='a_p',const=const_ex_138) #for excited state
    BaF.X.build_states(Fmax=9)
    BaF.A.build_states(Fmax=9)
    print(BaF)
    BaF.calc_branratios(threshold=0.0)
    
    #%% plotting spectra
    plt.figure('Spectra of two BaF isotopes')
    BaF.calc_spectrum(limits=(11627.0,11632.8))#11634.15,11634.36)#12260.5,12260.7)
    plt.plot(BaF.Eplt,BaF.I,label='$^{138}$BaF')
    plt.xlabel('transition frequency in 1/cm')
    plt.ylabel('intensity')
    plt.legend()
    
    print(BaF.X.const.show('non-zero'))
    #%% fermionic 137BaF
    BaF2 = Molecule(I1=1.5,I2=0.5,naturalabund=0.112,label='137 BaF',mass=mass137)
    BaF2.add_electronicstate('X',2,'Sigma',const=const_gr_137)
    BaF2.add_electronicstate('A',2,'Pi',Gamma=Gamma,const=const_ex_137)
    BaF2.X.build_states(Fmax=9.5)
    BaF2.A.build_states(Fmax=9.5)
    BaF2.calc_branratios()
    
    #%% plotting spectra with an offset
    BaF2.calc_spectrum(limits=(11627.0,11632.8))#11634.15,11634.36)#12260.5,12260.7)
    plt.plot(BaF2.Eplt,BaF2.I-10,label='$^{137}$BaF')
    plt.legend()
    
    #%% ground state Zeeman splitting due to external magnetic field in 138BaF
    BaF = Molecule(I1=0.5,naturalabund=0.717,label='138 BaF',verbose=False,mass=mass138)
    BaF.add_electronicstate('X',2,'Sigma', Hcase='b',const=const_gr_138) #for ground state
    BaF.add_electronicstate('A',2,'Pi',Gamma=Gamma, Hcase='a_p',const=const_ex_138) #for excited state
    BaF.X.build_states(Fmax=3,Fmin=0)
    BaF.A.build_states(Fmax=3,Fmin=0)
    
    BaF.X.plot_Zeeman(100e-4)
    
    #%% getting g-factors
    BaF = Molecule(I1=0.5,naturalabund=0.717,label='138 BaF',mass=mass138)
    BaF.add_electronicstate('X',2,'Sigma', Hcase='b',const=const_gr_138) #for ground state
    BaF.add_electronicstate('A',2,'Pi',Gamma=Gamma, Hcase='a_p',const=const_ex_138) #for excited state
    BaF.X.build_states(Fmax=4)
    BaF.X.get_gfactors()