"""pyOZ: An iterative Ornstein-Zernike equation solver """

import logging

from pyoz.core import System
from pyoz.potentials import *
from pyoz.properties import *
from pyoz import unit


__version__ = '0.4.0'
__author__ = 'Lubos Vrbka'


logging.basicConfig(filename='pyoz.log')
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
