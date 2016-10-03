import numpy as np
import pytest

import pyoz as oz


# The argument to all functions in this file, `two_component_lj`, is a solved
# two component Lennard-Jones system. It is defined by a pytest fixture in
# `conftest.py`.


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_kirkwood_buff(two_component_lj):
    kbi = oz.kirkwood_buff_integrals(two_component_lj)


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_isothermal_compressibility(two_component_lj):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_pressure_virial(two_component_lj):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_excess_chemical_potential(two_component_lj):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_two_particle_excess_entropy(two_component_lj):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_second_virial_coefficient(two_component_lj):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_activity_coefficient(two_component_lj):
    pass
