# -*- coding: utf-8 -*-
"""
Spectrum RaF
============

Calculating the optical spectrum of :math:`^{226}\\text{RaF}` using
spectroscopic constants.

Radium monofluoride is of particular interest due to its applications in
precision measurements with a heavy nucleus. 
"""

# sphinx_gallery_thumbnail_number = -1

from MoleCool.spectra import Molecule, ElectronicStateConstants, cm2THz
from MoleCool import pi, np, plt
from copy import deepcopy
# %%
# Spectroscopic constants
# -----------------------
# Literature:
# 
# - [a] Garcia Ruiz 2020: "Spectroscopy of short-lived radioactive molecules".
#   see Extended Data Table 2 in https://doi.org/10.1038/s41586-020-2299-4
#   using the relations :math:`\omega_e \chi_e\approx\omega_e^2/(4D_e)` and
#   :math:`A_e\approx A_{00}`.
# - [b] Udrescu 2023: "Precision spectroscopy and laser cooling scheme of a
#   radium-containing molecule" https://doi.org/10.1038/s41567-023-02296-w
#   using the relations :math:`B_e\approx 3/2\,B_0 - 1/2\, B_1` and
#   :math:`\alpha_e \approx B_0-B_1`.
# - [c] Petrov, Skripnikov 2020: "Energy levels of radium monofluoride RaF in
#   external electric and magnetic fields to search for P- and T,P-violation
#   effects" https://doi.org/10.1103/PhysRevA.102.062801
#   where the hyperfine tensor components are converted into Frosch and
#   Foley parameters.
# - [2] Skripnikov 2020: "Nuclear magnetization distribution effect in
#   molecules: Ra+and RaF hyperfine structure"
# - [5] Nuclear magnetic moments:
#   https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.59.771

X_cts = ElectronicStateConstants(
    const   = {
        'w_e'      : 441.8,         # [a]
        'w_e x_e'  : 1.6711,        # [a]
        'B_e'      : 192.5175e-3,   # [b]
        'D_e'      : 1.4e-7,        # [a]
        'alpha_e'  : 1.0650e-3,     # [b]
        'gamma'    : 5.8500e-3,     # [b]
        'b_F'      : 3.2133e-3,     # [c]
        'c'        : 0.6338e-3,     # [c]
        },
    unit    = '1/cm',
    )

A_cts = ElectronicStateConstants(
    const = {
        'T_e'       : 14321.377,    # [b]
        'w_e'       : 435.5,        # [a]
        'w_e x_e'   : 1.635,        # [a]
        'A_e'       : 2067.6000,    # [a]
        'p'         : -0.4107,      # [b]
        'B_e'       : 191.5375e-3,  # [b]
        'D_e'       : 1.4e-7,       # [a]
        'alpha_e'   : 1.0450e-3,    # [b]
        },
    unit    = '1/cm',
    )

# %%
# export all constants as dictionary:
X_cts.to_dict(exclude_default=False)

# Or as a nice format using pandas.DataFrame:
df = X_cts.show(
    formatting = 'non-zero',
    createHTML = 'cts_A', # to create a html file with nice format
    )

# which is the same as:
X_cts

# %%
# Setting up a molecule
# ---------------------
# The diatomic molecule :math:`^{226}\text{RaF}` has a single non-zero nuclear
# spin 1/2. We also set the temperature to 4 K which will affect the Boltzmann
# distribution of the calculated spectrum below.
# 
# We also add two electronic states -- the ground state `X` and first excited
# state `A` with respective Hund's cases `b` and `a_p` (meaning well-defined
# parity eigenstates due to lambda doubling).
# The vibrational states :math:`\nu` are set to 0.

mol = Molecule(
    I1      = 0.5,          # nuclear spin 1/2
    label   = '226 RaF',    # label
    mass    = 226+19,       # 
    temp    = 4,
    )
mol.add_electronicstate(
    'X', 2, 'Sigma',
    const   = X_cts,
    nu      = 0,
    Hcase   = 'b',
    )
mol.add_electronicstate(
    'A', 2, 'Pi',
    const   = A_cts,
    nu      = 0,
    Hcase   = 'a_p',
    Gamma   = 1/50e-9/(2*pi)*1e-6, # 50ns to in MHz [a]
    )

mol.build_states(Fmax = 4)

print(mol)
# %%
# or just a single electonic state:
print(mol.X)
# %%
# The constants object ``X_cts`` (which is displayed above) is now also stored
# as the attribute ``const`` inside the :class:`~.spectra.ElectronicState` object:
print(type(mol.A.const)) # mol.A.const is the same as A_cts

mol.A.const

# %%
# Electronic state
# ----------------
mol.X.get_eigenstates(
    mixed_states    = False,    # transform basis to good Hund's case basis
    rounded         = None,     # better overview in table entries
    createHTML      = False,    # to export html file
    )

# %%
# Zeeman splitting plot
# ^^^^^^^^^^^^^^^^^^^^^
mol.X.plot_Zeeman(np.linspace(0,70e-4,5))
plt.ylim(0.375,0.391)

# mol.X.get_gfactors()

# %%
# Optical spectra
# ---------------
mol.build_states(Fmax = 11)

mol.get_branratios(normalize = True)


# %%
# filtering transitions
# ^^^^^^^^^^^^^^^^^^^^^
# the index filter argument internally utilizes the filter function
# :func:`~.spectra.multiindex_filter` 

index_filter = (
    dict(N = 1),
    dict(P = +1, Om = 0.5, J = 0.5),
    )

mol.get_branratios(normalize    = False,
                   index_filter = index_filter)

# %%
# similarly the eigeneneries can be extracted:
df2 = mol.get_E(index_filter = index_filter, recalc_branratios = False)
f0  = df2.mean().mean()
print(f"Mean frequency of cooling transition: {f0*cm2THz:.6f} THz")
df2 

# mol.X.nu    = 1 
# f0_repump   = mol.get_E(index_filter = index_filter).mean().mean()
# print(f"Mean frequency of repumping transition: {f0*cm2THz:.6f} THz")

# %%
# Checking rotational leakage
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Branching ratios for the state A(Om = 1/2, J = 1/2^+) decaying in any ground
# state (in percent). They reveal a vanishing probability of population leakage
# in any other ground state than X(N=1).
    
mol.get_branratios(
    index_filter      = (dict(), dict(P = +1, Om = 0.5, J = 0.5)),
    recalc_branratios = False,
    ).round(6)*1e2

# %%
# Plotting a simple spectrum
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
mol.calc_branratios(threshold=0.0)
E, I = mol.calc_spectrum(
    limits = (f0-0.15,f0+0.1)
    )

plt.figure()
plt.plot(mol.Eplt, mol.I)
plt.xlabel('Frequency (cm$^{-1}$)')
plt.ylabel('Intensity (arb. u.)')


# %%
# Nice spectrum with fortrat diagram
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# function to convert the xaxis units from cm^-1 to GHz with an offset
def xconv(x):
    return (x * cm2THz - 398.258) * 1e3

# x-axis limits (always in units of cm^-1)
limits      = np.array([398.250, 398.260])/cm2THz

fig, axs    = plt.subplots(2,1, sharex=True)
ax          = axs[0]

### Plotting normal spectra
mol.calc_branratios(threshold=0.0) # calculate branching ratios with Boltzmann

for sigma, linewidth, color in zip([0, None], [0.8, 1.5], ['C0', 'k']):
    # calculate spectrum in a specific frequency range with Doppler broadening sigma
    E, I = mol.calc_spectrum(
        limits      = limits,
        sigma       = sigma,
        )
    # plotting the spectrum
    ax.plot(
        xconv(E), I/I.max(),
        lw = linewidth, color = color,
        )

### Plotting fortrat diagram

ax = axs[1]

# make a deep copy of the molecule as we modify the constants
molf = deepcopy(mol)
for const in [*ElectronicStateConstants.const_HFS, *ElectronicStateConstants.const_eq0Q]:    
    molf.X.const[const] = 0
molf.X.eigenst_already_calc = False
molf.calc_branratios(threshold=0.0) # calculate branching ratios with Boltzmann
molf.plot_fortrat(
    ["N","J"],
    ax              = ax,
    limits          = limits,
    limits_unit     = 'cm-1',
    branratio_TH    = 1e-2,
    legend          = True,
    xaxis_func      = xconv,
    )

# residual annotations and labels:
ax.set_ylim(top=11.9,bottom=-0.8)
ax.grid(True, axis='y')
ax.text(-0.1,6, "P$_1$", va='center', ha='right')
ax.text(1.3,7.8, "$^\\text{P}$Q$_{12}$", va='center', ha='right')
ax.text(-4.6,7.3, "$^\\text{Q}$R$_{12}$", va='bottom', ha='center')
ax.text(-5,5.2, "Q$_1$", va='top', ha='center')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Intensity (arb. u.)')

# %%
# Export properties for OBEs
# --------------------------

mol.X.Fmax  = 4
mol.A.Fmax  = 4

vibr_values = {
    "vibrbranch": [
        [0.996,     0.004],
        [0.004,     0.988],
        [1e-5,      0.008],
        [1e-6,      3e-5]], #[4]
    "vibrfreq": [
        [752.760,	729.040],
        [778.453,   753.113],
        [805.745,   778.628],
        [834.787,   805.715]],
    }

mol.export_OBE_properties(
    index_filter    = (dict(N=1), dict(P=+1,Om=0.5,J=0.5)),
    fname           = 'test_226RaF.json',
    HFfreq_offsets  = [0,0],
    QuNrs_const     = (['N'],['J']), # ([],[]),
    QuNrs_var       = ([],[]), # (['J','F'],['F'])
    include_mF      = False,
    vibr_values     = vibr_values,
    rounded         = 6,
    )