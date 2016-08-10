"""Iterative Ornstein-Zernike equation solver in Python. """

import logging

import simtk.unit as u
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB
import numpy as np

from pyoz.closure import supported_closures
from pyoz.solver import solve_ornstein_zernike

__all__ = ['solve_ornstein_zernike']

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


settings = dict()

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
settings['T'] = T
settings['epsilon_r'] = 78.3 * u.dimensionless
settings['epsilon_0'] = 8.854187817e-12 * u.coulomb**3 / u.joule / u.meter

# Coulomb interaction factor - Bjerrum length
# V(coul) in kT is then calculated as V = b_l * z1 * z2 / r
# with z in elementary charge units and r in A
settings['bjerrum_length'] = 0.0

kT = T * kB
settings['kB'] = kB
settings['kT'] = kT
settings['beta'] = 1 / kT
settings['e'] = (1 * u.elementary_charge).in_units_of(u.coulomb)

# Algorithm control
# =================

# number of discretization points
n_points_exp = 12
settings['n_points_exp'] = 12
n_points = 2 ** n_points_exp - 1
settings['n_points'] = n_points

dr = 0.05
settings['dr'] = dr

max_r = dr * n_points
settings['max_r'] = max_r
dk = np.pi / max_r
settings['dk'] = dk
settings['max_k'] = dk * n_points


settings['mix_param'] = 0.4
settings['tol'] = 1e-9
settings['max_iter'] = 1000
settings['max_dsqn'] = 100.0

# Potentials
# ==========

# Lennard-Jones
settings['lennard-jones'] = lj = dict()
lj['lj_sigma_rule'] = 'arithmetic'
lj['lj_epsilon_rule'] = 'geometric'

