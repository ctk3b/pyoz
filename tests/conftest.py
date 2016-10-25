import numpy as np
import pytest

import pyoz as oz


@pytest.fixture(scope='session')
def one_component_dpd():
    """Return a solved, unary DPD system. """
    T = 1
    rhos = 0.01

    dpd = oz.System(kT=T)

    r = dpd.r
    dpd.set_interaction(0, 0, oz.dpd(r, 10))
    dpd.solve(rhos=rhos, closure_name='hnc')
    return dpd


@pytest.fixture(scope='session')
def two_component_one_inf_dilute_dpd():
    """Return a solved, binary DPD system. """
    T = 1
    dpd = oz.System(kT=T)

    rhos = np.array([0.01, 0.0])
    r = dpd.r

    dpd.set_interaction(0, 0, oz.dpd(r, 10))
    dpd.set_interaction(0, 1, oz.dpd(r, 10))
    dpd.set_interaction(1, 1, oz.dpd(r, 10))

    dpd.solve(rhos=rhos, closure_name='hnc')
    return dpd


@pytest.fixture(scope='session')
def two_component_identical_dpd():
    """Return a solved, binary DPD system. """
    T = 1
    dpd = oz.System(kT=T)

    rhos = np.array([0.005, 0.005])


    r = dpd.r

    dpd.set_interaction(0, 0, oz.dpd(r, 10))
    dpd.set_interaction(0, 1, oz.dpd(r, 10))
    dpd.set_interaction(1, 1, oz.dpd(r, 10))


    dpd.solve(rhos=rhos, closure_name='hnc')
    return dpd


@pytest.fixture(scope='session')
def two_component_dpd():
    """Return a solved, binary DPD system. """
    T = 1
    rhos = np.array([0.01, 0.01])

    dpd = oz.System(kT=T)

    r = dpd.r
    dpd.set_interaction(0, 0, oz.dpd(r, 5))
    dpd.set_interaction(0, 1, oz.dpd(r, 10))
    dpd.set_interaction(1, 1, oz.dpd(r, 15))

    dpd.solve(rhos=rhos, closure_name='hnc')
    return dpd

