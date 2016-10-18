import numpy as np
import pytest

import pyoz as oz
from pyoz.exceptions import PyozError


# The argument to all functions in this file, `two_component_lj`, is a solved
# two component Lennard-Jones system. It is defined by a pytest fixture in
# `conftest.py`.


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_kirkwood_buff(two_component_lj):
    kbi = oz.kirkwood_buff_integrals(two_component_lj)


def test_structure_factor_single_component(one_component_lj):
    with pytest.raises(PyozError):
        oz.structure_factors(one_component_lj, formalism='fancy')

    sk_fz = oz.structure_factors(one_component_lj, formalism='faber-ziman')
    sk_al = oz.structure_factors(one_component_lj, formalism='ashcroft-langreth')
    assert np.array_equal(sk_fz[0, 0], sk_al[0, 0])


def test_structure_factor_multi_component(two_component_lj):

    sk_fz = oz.structure_factors(two_component_lj, formalism='faber-ziman')
    sk_al = oz.structure_factors(two_component_lj, formalism='ashcroft-langreth')
    assert np.array_equal(sk_fz[0, 0], sk_al[0, 0])
    assert np.array_equal(sk_fz[1, 1], sk_al[1, 1])
    assert np.array_equal(sk_fz[0, 1], sk_al[0, 1] + 1)
    assert np.array_equal(sk_fz[1, 0], sk_al[1, 0] + 1)


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_isothermal_compressibility(two_component_lj):
    pass


def test_pressure_virial(one_component_lj,
                         two_component_one_inf_dilute_lj,
                         two_component_identical_lj):
    P_one = oz.pressure_virial(one_component_lj)
    P_inf = oz.pressure_virial(two_component_one_inf_dilute_lj)
    P_two = oz.pressure_virial(two_component_identical_lj)

    assert np.allclose(P_one, P_inf, atol=1e-4)
    assert np.allclose(P_one, P_two, atol=1e-4)


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
