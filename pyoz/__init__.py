"""pyOZ: An iterative Ornstein-Zernike equation solver """

import logging


import numpy as np

from pyoz.core import System, Component
from pyoz.potentials import LennardJones, ContinuousPotential
from pyoz.thermodynamic_properties import (kirkwood_buff_integrals,
                                           excess_chemical_potential)
from pyoz import unit
import pyoz.unit as u
from pyoz.unit import BOLTZMANN_CONSTANT_kB as kB

__all__ = ['System', 'Component', 'unit',

           'ContinuousPotential', 'LennardJones',

           'kirkwood_buff_integrals', 'excess_chemical_potential']

__version__ = '0.4.0'
__author__ = 'Lubos Vrbka'


logging.basicConfig(filename='pyoz.log')
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info('pyOZ - version {:s}'.format(__version__))


defaults = dict()

# TODO: rework internal units to be consistent with the below
# Units
# =====
# distance: nm
# time: ps
# mass: amu
# charge: proton charge
# temperature: Kelvin
# angle: radians
# energy: kJ/mol

# Physical constants
# ==================
T = 300 * u.kelvin
defaults['T'] = T
defaults['epsilon_r'] = 78.3 * u.dimensionless
defaults['epsilon_0'] = 8.854187817e-12 * u.coulomb ** 3 / u.joule / u.meter

# Coulomb interaction factor - Bjerrum length
# V(coul) in kT is then calculated as V = b_l * z1 * z2 / r
# with z in elementary charge units and r in A
defaults['bjerrum_length'] = 0.0

kT = T * kB
defaults['kB'] = kB
defaults['kT'] = kT
defaults['beta'] = 1 / kT
defaults['e'] = (1 * u.elementary_charge).in_units_of(u.coulomb)

# Algorithm control
# =================

# number of discretization points
n_points_exp = 12
defaults['n_points_exp'] = 12
n_points = 2 ** n_points_exp
defaults['n_points'] = n_points

dr = 0.05 * u.angstrom
defaults['dr'] = dr

max_r = dr.value_in_unit(u.angstrom) * n_points
defaults['max_r'] = max_r
dk = np.pi / max_r
defaults['dk'] = dk
defaults['max_k'] = dk * n_points

defaults['iteration_scheme'] = 'picard'
defaults['mix_param'] = 1.0
defaults['tol'] = 1e-9
defaults['max_iter'] = 1000
defaults['max_dsqn'] = 100.0

# System info
# ===========
defaults['closure'] = 'hnc'
