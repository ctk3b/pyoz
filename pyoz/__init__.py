"""pyOZ: An iterative Ornstein-Zernike equation solver """

import logging

from pyoz.core import System, Component
from pyoz.potentials import (ContinuousPotential,
                             LennardJones,
                             Coulomb,
                             WCA)
from pyoz.thermodynamic_properties import (kirkwood_buff_integrals,
                                           excess_chemical_potential,
                                           pressure_virial,
                                           second_virial_coefficient)
from pyoz import unit

__all__ = ['System', 'Component', 'unit',

           'ContinuousPotential', 'LennardJones', 'Coulomb', 'WCA',

           'kirkwood_buff_integrals', 'excess_chemical_potential',
           'pressure_virial', 'second_virial_coefficient']

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
