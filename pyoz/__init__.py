"""pyOZ: An iterative Ornstein-Zernike equation solver """

import logging

from pyoz.core import System
from pyoz.potentials import *
from pyoz.thermodynamic_properties import (kirkwood_buff_integrals,
                                           excess_chemical_potential,
                                           pressure_virial,
                                           second_virial_coefficient,
                                           two_particle_excess_entropy)
from pyoz import unit

__all__ = ['System', 'Potential',

           'mie', 'wca', 'lennard_jones', 'coulomb', 'screened_coulomb',

           'kirkwood_buff_integrals', 'excess_chemical_potential',
           'pressure_virial', 'second_virial_coefficient',
           'two_particle_excess_entropy']

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
