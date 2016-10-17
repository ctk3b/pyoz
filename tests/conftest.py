import numpy as np
import pytest

import pyoz as oz
from pyoz.potentials import arithmetic, geometric


@pytest.fixture(scope='session')
def one_component_lj():
    """Return a solved, unary Lennard-Jones system. """
    T = 1
    sig = 1
    eps = 1
    rhos = 0.01

    lj = oz.System(kT=T)

    r = lj.r
    lj.set_interaction(0, 0, oz.lennard_jones(r, eps, sig))
    lj.solve(rhos=rhos, closure_name='hnc')
    return lj


@pytest.fixture(scope='session')
def two_component_one_inf_dilute_lj():
    """Return a solved, binary Lennard-Jones system. """
    T = 1
    sig = np.array([1, 2])
    eps = np.array([1, 0.75])
    eps_01 = geometric(eps[0], eps[1])
    sig_01 = arithmetic(sig[0], sig[1])
    rhos = np.array([0.01, 0.0]) / sig**3

    lj = oz.System(kT=T)

    r = lj.r
    lj.set_interaction(0, 0, oz.lennard_jones(r, eps[0], sig[0]))
    lj.set_interaction(1, 1, oz.lennard_jones(r, eps[1], sig[1]))
    lj.set_interaction(0, 1, oz.lennard_jones(r, eps_01, sig_01))

    lj.solve(rhos=rhos, closure_name='hnc')
    return lj


@pytest.fixture(scope='session')
def two_component_identical_lj():
    """Return a solved, binary Lennard-Jones system. """
    T = 1
    sig = np.array([1, 1])
    eps = np.array([1, 1])
    eps_01 = geometric(eps[0], eps[1])
    sig_01 = arithmetic(sig[0], sig[1])
    rhos = np.array([0.005, 0.005]) / sig**3

    lj = oz.System(kT=T)

    r = lj.r
    lj.set_interaction(0, 0, oz.lennard_jones(r, eps[0], sig[0]))
    lj.set_interaction(1, 1, oz.lennard_jones(r, eps[1], sig[1]))
    lj.set_interaction(0, 1, oz.lennard_jones(r, eps_01, sig_01))

    lj.solve(rhos=rhos, closure_name='hnc')
    return lj


@pytest.fixture(scope='session')
def two_component_lj():
    """Return a solved, binary Lennard-Jones system. """
    T = 1
    sig = np.array([1, 2])
    eps = np.array([1, 0.75])
    eps_01 = geometric(eps[0], eps[1])
    sig_01 = arithmetic(sig[0], sig[1])
    rhos = np.array([0.01, 0.01]) / sig**3

    lj = oz.System(kT=T)

    r = lj.r
    lj.set_interaction(0, 0, oz.lennard_jones(r, eps[0], sig[0]))
    lj.set_interaction(1, 1, oz.lennard_jones(r, eps[1], sig[1]))
    lj.set_interaction(0, 1, oz.lennard_jones(r, eps_01, sig_01))

    lj.solve(rhos=rhos, closure_name='hnc')
    return lj

