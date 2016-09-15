from collections import OrderedDict
from functools import partial

import numpy as np

import pyoz as oz
from pyoz.unit import AVOGADRO_CONSTANT_NA as Na
import pyoz.unit as u
from pyoz.scan import scan


def number_densities(colloid_diameter, SiO2_wt_perc):
    SiO2_mol_wt = 60.08 * u.gram / u.mole
    rho_solution = 1 * u.gram / u.centimeter**3
    colloid_rho = rho_solution * SiO2_wt_perc / SiO2_mol_wt * Na
    return d**3 * colloid_rho.in_units_of(u.nanometers**-3)  # Reduced units

def LR_parameters(colloid_diameter, NaCl_wt_perc):
    R = 8.314e-3 * u.kilojoules_per_mole / u.kelvin
    T = 300.0 * u.kelvin
    RT = R*T

    NaCl_mol_wt = 58.44 * u.gram / u.mole

    rho_solution = 1 * u.gram / u.centimeter**3

    # NaCl_conc = NaCl_wt_perc * 10.0 / NaCl_mol_wt  # mol/Liter
    NaCl_conc = rho_solution * NaCl_wt_perc / NaCl_mol_wt
    ionic_strength = NaCl_conc.value_in_unit(u.moles / u.liter)
    lambda_d = 0.304 / np.sqrt(ionic_strength) * u.nanometers
    lambda_b = 0.70 * u.nanometers


    l_debye = lambda_d / colloid_diameter  # Reduced units
    l_bjerrum = lambda_b / colloid_diameter  # Reduced units


    # Charge density of glass is given for a few different pH values:
    # http://physics.nyu.edu/grierlab/charge6c/
    q_density = 1e4 * u.elementary_charge / u.micrometers**2
    colloid_area = np.pi * colloid_diameter**2
    colloid_q = q_density * colloid_area

    #Equation 4: Bollinger, 2016 paper.
    # A/kT actually
    A = colloid_q**2 * l_bjerrum / (1 + 0.50 / l_debye)**2

    # TODO: Is there another unit reduction necessary here?
    A = A.value_in_unit(u.elementary_charge**2)
    return A, l_debye


# Create potential
def SALR(r, eps, m, n, LR_parms, **kwargs):
    SR = 4 * r**(-m)
    SA = -4 * eps * r**(-n)
    A, l_debye = LR_parms
    LR = A * np.exp(-(r - 1) / l_debye)
    return SR + SA + LR


if __name__ == '__main__':
    d = 40 * u.nanometers  # particle diameter
    d_SiO2_wt_perc = 1
    d_NaCl_wt_perc = 0.5
    d_epsilon = 1

    d_SiO2_wt_perc = 10
    d_NaCl_wt_perc = 5
    d_epsilon = 5

    SiO2_wt_perc = np.arange(5, 20, d_SiO2_wt_perc)
    rhos = [number_densities(colloid_diameter=d, SiO2_wt_perc=perc)
            for perc in SiO2_wt_perc]

    NaCl_wt_perc = np.arange(0.50, 6.25, d_NaCl_wt_perc)
    LR_parms = [LR_parameters(colloid_diameter=d, NaCl_wt_perc=perc)
                for perc in NaCl_wt_perc]

    epsilons = np.arange(1, 10, d_epsilon)

    # Variables to iterate over
    interactions = [(0, 0, partial(SALR, m=100, n=36))]
    variables = OrderedDict({'rho': rhos,
                             'eps': epsilons,
                             'LR_parms': LR_parms})

    # NOTE: Temperatures other than 298 K invalidate the Debye length
    # approximation used: kappa^-1 = 0.304 / sqrt(I(M))
    system_vars = {'T': 298,
                   'dr': 0.01,
                   'n_points': 8192}

    # Solve dat stuff
    thermo_data, raw_data = scan(variables, interactions, system_vars)

    # Store dat stuff
    ds = raw_data.to_dataset(name='RAW')
    ds.to_netcdf('data/SALR_raw_data.nc')

    ds = thermo_data.to_dataset(name='THERMO')
    ds.to_netcdf('data/SALR_thermo_data.nc')
