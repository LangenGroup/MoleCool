# -*- coding: utf-8 -*-
"""
====================================
Getting started
====================================

The user guide provides a more detailed walkthrough of MoleCool’s
features and usage.
"""

# %%
# ## Code structure
#
# ``MoleCool`` is built in an object-oriented fashion. The toolbox offers
# a streamlined workflow for initializing a System instance, enabling the
# setup of laser beams, magnetic fields, and customizable multi-level systems.
#
# For such configurations, the internal dynamics of a particle with complex
# light fields can be computed via the rate equations or the OBEs.
#
# The following diagram illustrates the code structure of the core classes.
#
# .. figure:: ../../_images/code_structure.jpg
#    :figwidth: 95 %
#    :align: center
#    :alt: code structure
#
#    Class diagram of the core classes and their relationships of ``MoleCool``.
#
# .. include:: ../../substitutions.rst

# %%
# ## System class
#
# The class |System| defines the central simulation environment.
# It stores all information about the laser setup, the magnetic
# field, the particle's level structure, position and velocity as well as
# computed quantities such as populations or trajectories.
#
# Creating the central interface for defining a physical setup typically
# starts with creating an object of |System|.

from MoleCool import System
system = System(description='my_first_test', load_constants='138BaF')

# %%
# Access its subsystems:

print(system.lasers)
print(system.levels)
print(system.Bfield)

# %%
# ## Lasersystem class
#
# The class |Lasersystem| defines the entire laser setup,
# consisting of multiple beams with optional frequency sidebands.
#
# The basic method ``add()`` constructs the laser system from single |Laser|
# instances. The method ``add_sidebands()`` is a wrapper for conveniently
# adding multiple frequency sidebands.

# single laser at 860 nm with linear polarization and 20 mW of power:
system.lasers.add(
    lamb=860e-9, P=20e-3, pol='lin',
)

# %%
# multiple laser objects with circular polarization are added at once:

system.lasers.add_sidebands(
    lamb=860e-9, P=10e-3, pol='sigmap',
    sidebands=[-1, 0, 1],
    offset_freq=20e6,
    mod_freq=40e6,
)
print(system.lasers)

# %%
# Each |Laser| object stores its parameters (power, wavelength, polarization,
# beam width, etc.) which can be accessed by indexing the |Lasersystem| object:

la = system.lasers[-1]
print(la)

# %%
# To remove lasers, use normal Python indexing:

del system.lasers[-1]
del system.lasers[:]

# %%
# ### Independent usage
#
# Each class can also be used independently for exploratory analysis:

from MoleCool import Lasersystem

lasers = Lasersystem()
lasers.add(lamb=860e-9, P=20e-3, pol='lin')

print(lasers)
lasers.plot_I_1D()

# %%
# or simply for calculating the intensity of a laser with given power and
# full width at half maximum (FWHM):

from MoleCool import Laser
print('Intensity in W/m^2:', Laser(P=20e-3, FWHM=5e-3).I)
print('Power in W:', Laser(I=700, FWHM=5e-3).P)

# %%
# ## Levelsystem class
#
# The class |Levelsystem| organizes all electronic (|ElectronicState|)
# and other quantum states (|State|), allowing customizable and
# interactive modifications of multi-level systems and their properties.

from MoleCool import Levelsystem

# initiate empty level system
levels = Levelsystem()

# add electronic ground ('gs') and excited ('exs') state
levels.add_electronicstate('S12', 'gs')
levels.add_electronicstate('P32', 'exs', Gamma=6.065)

# %%
# add single quantum states:

levels.S12.add(J=1/2, F=[1, 2])
levels.P32.add(J=3/2, F=[0, 1, 2, 3])

# %%
# print all defined states with their quantum numbers:

print(levels)
print(levels.S12)
print(levels.S12[0])

# %%
# Specific electronic or quantum states can also be removed:

levels.add_electronicstate('D52', 'exs', Gamma=1.0)
del levels['D52']
del levels.P32[0]

# %%
# States can only be added or deleted before any level-system property is initialized.
#
# For an empty level system, these properties are automatically generated
# and can be modified using pandas DataFrame objects.

levels.wavelengths.loc[('S12'), ('P32')] = 780.241
levels.dMat_red.loc[('S12', 1/2, 2), ('P32', 3/2, 0)] = 0
print(levels.dMat_red)

# %%
# modify g-factors and frequencies:

levels.S12.gfac.iloc[0] = 0.1234
levels.S12.freq.iloc[0] = -4.272
levels.P32.freq.loc[('P32', 3/2, 1)] = 10.
print(levels.P32.freq)
levels.print_properties()

# %%
# Properties can also be imported from a `.json` file created by the spectra module:

system = System(description='my_first_test', load_constants='138BaF')
system.levels.add_electronicstate('X', 'gs')
system.levels.add_electronicstate('A', 'exs')

system.levels.X.load_states(v=[0, 1])
system.levels.A.load_states(v=[0])
print(system.levels)

# %%
# ## Bfield module
#
# The class |Bfield| describes the magnetic field.
# Every |System| instance contains a default zero field.
# A static, uniform field with strength 5 G and angle 60° relative to z-axis:

system.Bfield.turnon(
    strength=5e-4,
    direction=[0, 0, 1],
    angle=60
)

# %%
# The magnetic field can be reset to zero at any time with `system.Bfield.reset()`.

# %%
# ## Internal dynamics computation
#
# MoleCool uses `scipy.integrate.solve_ivp()` to solve ODEs and JIT-compiles them
# with numba for near-C performance.

# %%
# ## Example: Optical cycling simulation for BaF

from MoleCool import System
import numpy as np

system = System(description='SimpleTest1_BaF', load_constants='138BaF')

# %%
# Define lasers with multiple sidebands:

for lamb in np.array([859.830, 895.699, 897.961]) * 1e-9:
    system.lasers.add_sidebands(lamb=lamb, P=20e-3, pol='lin',
                                offset_freq=19e6, mod_freq=39.33e6,
                                sidebands=[-2, -1, 1, 2],
                                ratios=[0.8, 1, 1, 0.8])

# %%
# Load all relevant vibrational states:

system.levels.add_all_levels(v_max=2)

# %%
# Run a rate-equation simulation:

system.calc_rateeqs(t_int=20e-6)

# %%
# Plot populations and forces:

system.plot_N()
system.plot_F()
system.plot_Nscatt()

# %%
# ## Other modules
#
# ### spectra module
#
# The module :mod:`~MoleCool.spectra` provides an independent toolset for
# computing molecular spectra and exporting constants to `.json` files.

from MoleCool.spectra import ElectronicStateConstants, Molecule, plt

const_gr = ElectronicStateConstants(const={
    'B_e': 0.2159, 'D_e': 1.85e-7, 'gamma': 0.0027,
    'b_F': 0.0022, 'c': 0.00027,
})
const_ex = ElectronicStateConstants(const={
    'B_e': 0.2117, 'D_e': 2.0e-7, 'A_e': 632.2818,
    'p': -0.089545, 'q': -0.0840,
    "g'_l": -0.536, "g'_L": 0.980,
    'T_e': 11946.31676963,
})

# %%
# Initiating a Molecule and adding electronic states:

BaF = Molecule(I1=0.5, mass=157, temp=4)
BaF.add_electronicstate('X', 2, 'Sigma', const=const_gr)
BaF.add_electronicstate('A', 2, 'Pi', Gamma=2.84, const=const_ex)
BaF.build_states(Fmax=8)

# %%
# Calculating branching ratios and molecular spectrum:

BaF.calc_branratios()
E, I = BaF.calc_spectrum(limits=(11627.0, 11632.8))

# %%
# Plotting the spectrum:

plt.figure()
plt.plot(E, I)
plt.xlabel('Frequency (cm$^{-1}$)')
plt.ylabel('Intensity (arb. u.)')
