import numpy as np
import pytest

import pyoz as oz
from pyoz.potentials import arithmetic, geometric


@pytest.fixture(scope='session')
def two_component_lj():
    T = 1
    sig = np.array([1, 2])
    eps = np.array([1, 0.75])
    eps_01 = geometric(eps[0], eps[1])
    sig_01 = arithmetic(sig[0], sig[1])
    rhos = np.array([0.01, 0.01]) / sig**3

    lj = oz.System(kT=T)

    r = lj.r
    lj.set_interaction(0, 0, oz.lennard_jones(r, eps[0] / T, sig[0]))
    lj.set_interaction(1, 1, oz.lennard_jones(r, eps[1] / T, sig[1]))
    lj.set_interaction(0, 1, oz.lennard_jones(r, eps_01 / T, sig_01))

    lj.solve(rhos=rhos, closure_name='hnc')
    return lj

