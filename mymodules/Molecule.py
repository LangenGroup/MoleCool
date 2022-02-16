# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:03:28 2021

@author: Felix

v0.2.4

Module for calculating the eigenenergies and eigenstates of diatomic molecules
exposed to external fields.
Therefore molecular constants which are measured and fitted in spectroscopic
experiments must be provided to build up the effective Hamiltonian terms.
Finally, the transition probabilities between two given electronic
state manifolds can be determined to simulate a complete spetrum.

Example
-------

Example for calculating and plotting a spectrum of 138BaF::

    from Molecule import *    
    # molecular constants in wave numbers (1/cm) within the effective Hamiltonian
    const_gr_138 = ElectronicStateConstants(const={
                    'B_e':0.21594802,'D_e':1.85e-7,'gamma':0.00269930,
                    'b_F':0.002209862,'c':0.000274323})
    const_ex_138 = ElectronicStateConstants(const={
                    'A':632.28175,'A_D':3.1e-5,'B_e':0.2117414,
                    'D_e':2.0e-7,'p+2q':-0.25755,"g'_l":-0.536,"g'_L":0.980})
    
    # create a Molecule instance with nuclear spin 1/2
    BaF = Molecule(I1=0.5,transfreq=11946.31676963)
    
    # add a ground and an excited electronic state and build up hyperfine states
    BaF.add_electronicstate('X',2,'Sigma', const=const_gr_138)
    BaF.add_electronicstate('A',2,'Pi', const=const_ex_138)
    BaF.X.build_states(Fmax=9)
    BaF.A.build_states(Fmax=9)
    
    # calculate branching ratios and plot spectrum
    BaF.calc_branratios(threshold=0.05)
    BaF.calc_spectrum( limits=(11627.0, 11632.8) )
    plt.plot(BaF.Eplt, BaF.I)
    plt.xlabel('transition frequency in 1/cm')
    plt.ylabel('intensity')
    
Tip
---
The instances of all classes :class:`~Molecule.Molecule`,
:class:`ElectronicState`, :class:`Hcasea` and :class:`ElectronicStateConstants`
can be printed::
    
    print(BaF)
    print(BaF.X)
    print(BaF.X.states[0])
    print(const_gr_138) # is the same as: print(BaF.X.const)
"""
from System import *
from collections.abc import Iterable
#: Constant for converting a unit in wavenumbers (cm^-1) into MHz.
cm2MHz = 299792458.0*100*1e-6 #using scipys value for the speed of light
#%% classes
class Molecule:
    def __init__(self,I1=0,I2=0,Bfield=0.0,mass=None,load_constants=None,
                 temp=5.0,Gamma=21.0,naturalabund=1.0,transfreq=0.0,
                 label='BaF',verbose=True):        
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
            mass of the molecule. The default is None.
        load_constants : string, optional
            if provided the molecular constants are loaded from an external
            file. The default is None.
        temp : float, optional
            temperatur of the molecule in K. The default is 5.0.
        Gamma : float, optional
            natural linewidth of the excited state as an angular frequency
            (2pi * MHz). The default is 21.0.
        naturalabund : float, optional
            natural abundance in the range from 0.0 to 1.0 of different isotopes
            of a molecule in order to weight the spectra for different isotopes.
            The default is 1.0.
        transfreq : float, optional
            transition frequency between the ground and excited state in units
            of wavenumbers 1/cm. The default is 0.0. This value `transfreq` is
            used as offset for the calculation of the spectrum but if the constants
            `T_e` or `w_e` in :class:`ElectronicStateConstants` are non-zero,
            these values are used instead to calculate the energy offset
            (see :meth:`ElectronicStateConstants.electrovibr_energy`).
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
        cm2MHz = c*100*1e-6
        # these two values are actually dependent on the electronic state!??
        self.Gamma      = Gamma/cm2MHz #in MHz (without 2pi) and then to cm^-1
        self.transfreq  = transfreq #in cm^-1
        self.naturalabund = naturalabund
        self.label      = label
        self.verbose    = verbose
        self.grstates   = [] #empty lists to hold labels of electronic states
        self.exstates   = [] #to be appended later
        if load_constants: #or better in Electronic state???
            #maybe call other function to import molecular constants from a file
            pass
        #elements, isotopes, natural abundance, nuclear spin magn moment, ..
        
    
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
        self.__dict__[args[0]].verbose = self.verbose
        if self.__dict__[args[0]].grex == 'ground state':
            self.grstates.append(args[0])
        else: self.exstates.append(args[0])
    
    def calc_branratios(self,threshold=0.05,E_lowest=0.0,include_Boltzmann=True,
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
            all branching ratios below the threshold  in the range from 0.0
            to 1.0 are set to zero. The default is 0.05.
        E_lowest : float, optional
            if the Boltzmann factor is included the respective energy difference
            is calculated with respect to the energy `E_lowest`.
            However it is ensured that no eigenenergy of any of the ground state
            levels is below this energy `E_lowest`. The default is 0.0.
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
        # transition energies
        E_vibX, E_vibA = X.const.electrovibr_energy, A.const.electrovibr_energy
        if (E_vibX == 0.0) and (E_vibA == 0.0):
            E_offset = self.transfreq
        else:
            E_offset = E_vibA - E_vibX
        self.E = A.Ew[None,:] - X.Ew[:,None] + E_offset
        
        if include_Boltzmann:
            cm2MHz = c*100*1e-6
            fac = 1*cm2MHz*1e6*h/k_B
            if np.min(X.Ew) < E_lowest: E_lowest = np.min(X.Ew)
            Boltzmannfac = np.exp(-(X.Ew-E_lowest)[:,None]/fac/self.temp)
        else:
            Boltzmannfac = 1.0
        self.branratios = self.dipmat**2 * Boltzmannfac
        # additional degeneracy factor here?
        
        #set all branching ratios smaller than the threshold (default 0.05) to zero
        self.branratios = np.where(self.branratios<=threshold,0.0,self.branratios)
    
    def calc_spectrum(self,limits=None,plotpoints=40000):
        """ calculates the spectrum in a certain frequency range using the
        branching ratios previously calculated in the method :func:`calc_branratios`.
        The resulting frequency and intensity arrays are not only returned but
        also stored as variables ``Eplt`` and ``I`` in the :class:`~Molecule.Molecule`
        instance. The widths of the single transitions are determined by the
        natural linewidth ``Gamma`` of the respective :class:`~Molecule.Molecule` instance.

        Parameters
        ----------
        limits : tuple, optional
            defines the frequency limits for the plotting the spectrum as list
            or tuple of size 2 in units of wavenumbers 1/cm. By default the
            complete range containing all transitions is chosen. 
            The default is None.
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
        Gamma       = self.Gamma
        if limits == None: #limits of the energiy range for the spectrum to be plotted
            Emin,Emax   = np.min(self.E)-0.1 , np.max(self.E)+0.1
        else:
            Emin,Emax = limits
        Eplt        = np.linspace(Emin,Emax,plotpoints)
        I           = np.zeros(Eplt.size)
        for i in range(X.N):
            for j in range(A.N):
                branratio = self.branratios[i,j]
                if branratio == 0.0: continue
                I += branratio / ((self.E[i,j]-Eplt)**2+(Gamma/2)**2)
        I *= self.naturalabund*Gamma/(2*pi)
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
            print('lower eigenstate {:3} & upper eigenstate {:3}, branratio {:3.0f}%, energy {:.7e}'.format(
                st_l,st_u,self.branratios[st_l,st_u]*100,self.E[st_l,st_u]))
        # for st_l in np.unique(st_l_arr):
        #     print(self.X.Ev[:,st_l])
        
    def __str__(self):
        """prints all general information of the Molecule with its electronic states"""
        
        str1 = 'Molecule {}: with the nuclear spins I1 = {}, I2 = {} and mass {}'.format(
            self.label,self.I1,self.I2,self.mass)
        str1+= '\n magnetic field strength: {:.2e}G, temperature: {:.2f}K'.format(
            self.Bfield*1e4,self.temp)
        str1+= '\nIncluding the following defined electronic states:\n'
        for state in [*self.grstates,*self.exstates]:
            str1+='* {}\n'.format(self.__dict__[state])
            
        cm2MHz = c*100*1e-6
        # these two values are actually dependent on the electronic state!??
        self.Gamma #in MHz (without 2pi) and then to cm^-1
        # self.transfreq #in cm^-1
        return str1
#%%
class ElectronicStateConstants: 
    #: electronic energy offset constant
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
                       'a_2','b_F_2','c_2'] #for the second nuclear spin if I2 is non-zero
    #: electric quadrupol interaction
    const_eq0Q      = ['eq0Q']
    #: Lambda-doubling constants
    const_LD        = ['o+p+q','p+2q','q','(p+2q)_D','q_D']
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
        via the function :func:`~System.save_object` of the :mod:`System` module
        and can be loaded later using the function :func:`~System.open_object`.
        
        Raises
        ------
        KeyError
            if the dictionary `const` or the keyword arguments `kwargs`
            contains some values for which the respective key is not defined.
        """
        units = ['1/cm','MHz']
        cm2MHz = c*100*1e-6 # convert from 1/cm into MHz
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
        cm2MHz = c*100*1e-6
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
                          columns=['value [1/cm]','value [MHz]'])
        
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

    def to_dict(self):
        """Converts the defined constants to a dictionary which also includes
        the calculated values :meth:`B_v` and :meth:`D_v`.

        Returns
        -------
        dic : dict
            dictionary with all defined constants.
        """
        dic = {key : self.__dict__[key] for key in self.const_all}
        dic['A_v'] = self.A_v
        dic['B_v'] = self.B_v
        dic['D_v'] = self.D_v
        return dic

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
    def __init__(self,label,Smultipl,L,Hcase='a',nu=0,const={}):
        """This class represents an electronic ground or excited state manifold
        which are part of the molecular level structure.
        After an electronic state is created with certain constants of the effective
        Hamiltonian all the single hyperfine states can be added in order to
        calculate the eigenstates and eigenenergies (see :meth:`calc_eigenstates`
        and :meth:`get_eigenstates`).
        
        Parameters
        ----------
        label : string
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
        Hcase : string, optional
            Hund's case describing the states within the electronic
            state manifold. The default is 'a'.
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
        self.Hcase      = Hcase
        #constants
        if type(const) == ElectronicStateConstants:
            self.const = const
            self.const.nu = nu
        else:
            self.const = ElectronicStateConstants(const=const,nu=nu)
        
        # vibrational level
        self.nu         = nu #have to be called after init of self.const
        #self.parity or symmetry for Sigma states --> + or -
        if self.S > 0:  self.shell = 'open'
        else:           self.shell = 'closed'
        #: list for storing the pure states which can be added after class initialization
        self.states = []
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
        if Omega > self.L: pm = +1
        else: pm = -1
        
        E = cs.electrovibr_energy + pm*A_v*self.L*self.S \
            + (B_v *(J*(J+1) + self.S*(self.S+1) - Omega**2 - self.S**2) - D_v *(J*(J+1))**2) \
            + p*phs(J+0.5) * cs['p+2q']/2 *(J+0.5)
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
                            if self.Bfield != 0.0:
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
                                # if J < (abs(Om)-1e-3): continue # also here this selection rule???
                                if self.Bfield != 0.0:
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
        
    def get_eigenstates(self,precision=4,onlygoodQuNrs=True,createHTML=False):
        """
        returns the sorted eigenenergies and respective eigenstates determined
        by the method :func:`calc_eigenstates` in a nice format via the datatype
        `pandas.DataFrame` in order to be printed.

        Parameters
        ----------
        precision : int, optional
            precision to which the values of the eigenstates are rounded.
            The default is 4.
        onlygoodQuNrs : bool, optional
            specifies if only the good Quantum numbers are included for getting
            a better overview of the printed DataFrame. The default is True.
        createHTML : bool, optional
            if True a Html file `eigenstates.html` with the DataFrame is generated
            for a better view of the eigenstates. The default is False.

        Returns
        -------
        pandas.DataFrame
            the rounded DataFrame comprising the eigenvalues and eigenstates to
            be nicely printed
        """
        pd.options.display.max_columns = None   # in order to display all eigenstates
        pd.options.display.max_rows = None      # even if there are a lot
        # pd.set_option("display.precision", precision)
        #pd.reset_option('display')
        if not self.eigenst_already_calc: self.calc_eigenstates()
        
        col_arr = [np.arange(len(self.Ew)),np.around(self.Ew,precision)]
        DF1 = pd.DataFrame(self.Ev, 
                           index=pd.MultiIndex.from_frame(
                               self.get_purestates(onlygoodQuNrs=onlygoodQuNrs)),
                           columns=pd.MultiIndex.from_arrays(col_arr,names=('eigenvector i','eigenvalue')))
        DF1 = DF1.round(precision)
        if createHTML:
            #render dataframe as html
            html = DF1.to_html()
            #write html to file
            text_file = open("eigenstates.html", "w")
            text_file.write(html)
            text_file.close()
        else:
            return DF1
        
    def get_purestates(self,onlygoodQuNrs=False):
        """
        returns the pure states included in the instance :class:`ElectronicState`
        in a nice format via the datatype `pandas.DataFrame` in order to be printed.
        But at first any states have to be added via :func:`build_states`.

        Parameters
        ----------
        onlygoodQuNrs : bool, optional
            specifies if only the good Quantum numbers are included for getting
            a better overview of the printed DataFrame. The default is False.

        Returns
        -------
        pandas.DataFrame
            the rounded DataFrame comprising the pure states to be nicely printed
        """
        for i,st in enumerate(self.states):
            if i==0: DF = st.DF(onlygoodQuNrs)
            else: DF = DF.append(st.DF(onlygoodQuNrs),ignore_index=True)
        return DF
    
    def get_gfactors(self,Bmax=0.01e-4):
        """calculates the mixed g-factors for every hyperfine level eigenstate.
        These g-factors are returned as an array with the same order as the
        eigenstates for zero magnetic field which can be printed via 
        :meth:`get_eigenstates`. The gfactor array is also stored in the
        attribute ``gfactors``.

        Parameters
        ----------
        Bmax : float, optional
            maximum magnetic field strength in T to which the mixed g-factors are
            calculated. This value should be in the small region where the Zeeman
            shifts possess only linear behavior. The default is 0.01e-4.

        Returns
        -------
        numpy.ndarray
            array containing the mixed g-factors ordered by energy of the eigenstates.
        """
        oldBfield = self.Bfield
        oldstates = self.states
        if oldBfield == 0.0:
            self.Bfield = 1e-5 #set to arbitrary non-zero value
            self.build_states(self.Fmax,self.Fmin)
        
        N           = self.N
        Bfield      = np.array([1e-3*Bmax,Bmax])
        Ew_B_arr    = np.zeros((len(Bfield),N))
        for k,B in enumerate(Bfield): #maybe use for the zero Bfield energies Bfield=0.0? --> is also much faster
            self.Bfield = B
            self.calc_eigenstates()
            Ew_B_arr[k,:] = self.Ew
        
        mu_B,cm2MHz = physical_constants['Bohr magneton'][0], c*100*1e-6
        dB          = Bfield[1] - Bfield[0]
        gfactors    = np.zeros(N)
        mF_arr      = np.zeros(N)
        for i,vec in enumerate(self.Ev.T):
            j           = np.argsort(np.abs(vec))[-1]
            mF          = self.states[j].mF
            mF_arr[i]   = mF
            if mF == 0.0:
                continue
            gfactors[i] = (Ew_B_arr[1,i] - Ew_B_arr[0,i])*cm2MHz*1e6*h/(mu_B*dB*mF)
        
        gfactors_red = []
        i = 0
        for tmp in range(N):
            gfactors_red.append(gfactors[i])
            if mF_arr[i] == 0.0:
                i += 1
            else: 
                i += int(2*abs(mF_arr[i]) + 1 +0.01)
            if i >= N: break
        self.gfactors = np.array(gfactors_red)
        
        # set previous value to the Bfield variable
        self.Bfield = oldBfield
        self.build_states(Fmax=self.Fmax,Fmin=self.Fmin)
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
        
        oldBfield = self.Bfield
        if oldBfield == 0.0:
            self.Bfield = 1e-5 #set to arbitrary non-zero value
            self.build_states(self.Fmax,self.Fmin)
        Ew_B_arr = np.zeros((len(Bfield),self.N))
        for k,B in enumerate(Bfield):
            self.Bfield = B
            self.calc_eigenstates()
            Ew_B_arr[k,:] = self.Ew
        
        # plotting 
        Ew_B_arr = eigensort(Bfield,Ew_B_arr)
        plt.figure('Zeeman splitting')
        for i in range(Ew_B_arr.shape[1]):
            plt.plot(Bfield*1e4,Ew_B_arr[:,i],'-')
        plt.xlabel('magnetic field in G')
        plt.ylabel('Energy in cm$^{-1}$')
        
        # set previous value to the Bfield variable
        self.Bfield = oldBfield
        
    def __str__(self):
        """prints all general information of the ElectronicState"""
        Lnames = ['Sigma','Pi','Delta','Phi','Gamma']
        return '{:13s} {}(^{}){} - Hunds case {} - {} shell electronic state'.format(
            self.grex,self.label,self.Smultipl,Lnames[self.L],self.Hcase,self.shell) \
            + '\n  --> includes {} states'.format(self.N)

    @property
    def nu(self):
        """vibrational quantum number for the vibrational levels.
        Can be called and changed.
        """
        return self.__nu # maybe check if self.const.nu is different?
    
    @nu.setter
    def nu(self,val):
        if not isint(val):
            raise ValueError('given value {} is no integer!'.format(val))
        self.const.nu = val
        self.__nu = val
        
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
class Hcasea:
    goodQuNrs = ['L','Si','Om','J','F'] #different for Fermions and bosons?
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
    def __str__(self):
        print(self.DF())
        return 'Hunds case a'
    
    def DF(self,onlygoodQuNrs=False):
        if onlygoodQuNrs:
            QuNrs = self.goodQuNrs
        else: QuNrs = self.QuNrs
        return pd.DataFrame([[self.__dict__[Nr] for Nr in QuNrs]],columns=QuNrs)
    # maybe new hunds case for two nuclear spins with an option: stron coupling I1 and weak I2 =True
    def to_Hcaseb(self):
        # return Hcaseb(quNrs,states=[],prefactors=[])
        prefacs = []
        for N in addJ(self.S,self.J):
            prefac = np.sqrt(2*N+1)*phs(self.J+self.Om)*w3j(self.S,N,self.J,self.Si,self.L,-self.Om)
            print('{:+.4f} * |N={}> '.format(prefac,N),end=' ')
    
# class Eigenstates: #??
#     def __init__(self):
#         pass
        

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
        H_hfs2 = kd(L,L_)*phs(F1_+I2+F)*phs(J+I1+F1_+1)*cb(I2)*sb(F1)*sb(F1_)*sb(J)*sb(J_) \
                *w6j(I2,F1_,F,F1,I2,1)*w6j(J_,F1_,I1,F1,J,1) * sum1
        
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
    if ((const['o+p+q']!=0.0) or (const['p+2q']!=0.0) or (const['q']!=0.0)) and (kd(J,J_) != 0.0):
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
                
            sum1 += kd(L,L_-2*q)*(  const['o+p+q'] * kd(Om,Om_)*sum11 
                                  + const['p+2q'] * sum12
                                  + const['q'] * kd(Si,Si_)*sum13   )
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
def w3j(j_1, j_2, j_3, m_1, m_2, m_3):
    """returns Wigner 3j-symbol with arguments (j_1, j_2, j_3, m_1, m_2, m_3)"""
    return float(wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3))
def w6j(j_1, j_2, j_3, j_4, j_5, j_6):
    """returns Wigner 6j-symbol with arguments (j_1, j_2, j_3, j_4, j_5, j_6)"""
    return float(wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6))

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

#%%
if __name__ == '__main__':
    cm2MHz = 29979.2458 # actually c*100*1e-6 ?
    # BaF constants for the isotopes 138 and 137 from Steimle paper 2011
    const_gr_138 = {'B_e':0.21594802,'D_e':1.85e-7,'gamma':0.00269930,
                    'b_F':0.002209862,'c':0.000274323} #,'g_l':-0.028}#gl constant??
    const_ex_138 = {'A_e':632.28175,'A_D':3.1e-5,
                    'B_e':0.2117414, 'D_e':2.0e-7,'p+2q':-0.25755,"g'_l":-0.536,"g'_L":0.980}
    
    const_gr_137 = {'B_e':0.21613878,'D_e':1.85e-7,'gamma':0.002702703,
                    'b_F':0.077587, 'c':0.00250173,'eq0Q':-0.00390270*2,
                    'b_F_2':0.002209873,'c_2':0.000274323}
    const_ex_137 = {'B_e':0.211937,'D_e':2e-7,'A_e':632.2802,'A_D':3.1e-5,
                    'p+2q':-0.2581,'d':0.0076}
    
    #%% bosonic 138BaF
    BaF = Molecule(I1=0.5,transfreq=11946.31676963,naturalabund=0.717,
                    Gamma=21.0,label='138 BaF')
    BaF.add_electronicstate('X',2,'Sigma', const=const_gr_138)
    BaF.add_electronicstate('A',2,'Pi', const=const_ex_138)
    BaF.X.build_states(Fmax=9)
    BaF.A.build_states(Fmax=9)
    BaF.calc_branratios(threshold=0.05)
    
    #%% plotting spectra with an offset
    plt.figure('Spectra of two BaF isotopes')
    BaF.calc_spectrum(limits=(11627.0,11632.8))#11634.15,11634.36)#12260.5,12260.7)
    plt.plot(BaF.Eplt, BaF.I+800,label='$^{138}$BaF')
    plt.xlabel('transition frequency in 1/cm')
    plt.ylabel('intensity')
    plt.legend()
    
    print(BaF.X.const.show('non-zero'))
    #%% fermionic 137BaF
    BaF2 = Molecule(I1=1.5,I2=0.5,transfreq=11946.3152748,naturalabund=0.112,
                    Gamma=21.,label='137 BaF')
    BaF2.add_electronicstate('X',2,'Sigma',const=const_gr_137)
    BaF2.add_electronicstate('A',2,'Pi',const=const_ex_137)
    BaF2.X.build_states(Fmax=9.5)
    BaF2.A.build_states(Fmax=9.5)
    BaF2.calc_branratios(threshold=0.05)
    
    #%% plotting spectra with an offset
    BaF2.calc_spectrum(limits=(11627.0,11632.8))#11634.15,11634.36)#12260.5,12260.7)
    plt.plot(BaF2.Eplt,BaF2.I+600,label='$^{137}$BaF')
    plt.legend()
    
    #%% ground state Zeeman splitting due to external magnetic field in 138BaF
    BaF = Molecule(I1=0.5,transfreq=11946.31676963,naturalabund=0.717,
                    Gamma=21.0,label='138 BaF',verbose=False)
    BaF.add_electronicstate('X',2,'Sigma',const=const_gr_138) #for ground state
    BaF.add_electronicstate('A',2,'Pi',const=const_ex_138) #for excited state
    BaF.X.build_states(Fmax=3,Fmin=0)
    BaF.A.build_states(Fmax=3,Fmin=0)
    
    BaF.X.plot_Zeeman(100e-4)
    
    #%% getting g-factors
    BaF = Molecule(I1=0.5,transfreq=11946.31676963,naturalabund=0.717,
                   Gamma=21.0,label='138 BaF')
    BaF.add_electronicstate('X',2,'Sigma', const=const_gr_138)
    BaF.add_electronicstate('A',2,'Pi', const=const_ex_138)
    BaF.X.build_states(Fmax=4)
    BaF.X.get_gfactors()