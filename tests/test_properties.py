import numpy as np
import pytest

import pyoz as oz
from pyoz.exceptions import PyozError


# The argument to all functions in this file, `two_component_dpd`, is a solved
# two component Lennard-Jones system. It is defined by a pytest fixture in
# `conftest.py`.


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_kirkwood_buff(two_component_dpd):
    kbi = oz.kirkwood_buff_integrals(two_component_dpd)


def test_structure_factor_single_component(one_component_dpd):
    with pytest.raises(PyozError):
        oz.structure_factors(one_component_dpd, formalism='fancy')

    sk_fz = oz.structure_factors(one_component_dpd, formalism='faber-ziman')
    sk_al = oz.structure_factors(one_component_dpd, formalism='ashcroft-langreth')
    assert np.array_equal(sk_fz[0, 0], sk_al[0, 0])


def test_structure_factor_multi_component(two_component_dpd):

    sk_fz = oz.structure_factors(two_component_dpd, formalism='faber-ziman')
    sk_al = oz.structure_factors(two_component_dpd, formalism='ashcroft-langreth')
    assert np.array_equal(sk_fz[0, 0], sk_al[0, 0])
    assert np.array_equal(sk_fz[1, 1], sk_al[1, 1])
    assert np.array_equal(sk_fz[0, 1], sk_al[0, 1] + 1)
    assert np.array_equal(sk_fz[1, 0], sk_al[1, 0] + 1)


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_isothermal_compressibility(two_component_dpd):
    pass


def test_pressure_virial(one_component_dpd,
                         two_component_one_inf_dilute_dpd,
                         two_component_identical_dpd):
    P_one = oz.pressure_virial(one_component_dpd)
    P_inf = oz.pressure_virial(two_component_one_inf_dilute_dpd)
    P_two = oz.pressure_virial(two_component_identical_dpd)

    assert np.allclose(P_one, P_inf, atol=1e-4)
    assert np.allclose(P_one, P_two, atol=1e-4)


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_excess_chemical_potential(two_component_dpd):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_two_particle_excess_entropy(two_component_dpd):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_second_virial_coefficient(two_component_dpd):
    pass


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_activity_coefficient(two_component_dpd):
    pass


def test_internal_energy(one_component_dpd,
                         two_component_one_inf_dilute_dpd,
                         two_component_identical_dpd):

    U1 = oz.internal_energy(one_component_dpd)
    U1_pair = oz.internal_energy(one_component_dpd, pair=(0, 0))
    U2_inf = oz.internal_energy(two_component_one_inf_dilute_dpd)
    U2_inf_pair = oz.internal_energy(two_component_one_inf_dilute_dpd, pair=(0, 0))
    U2 = oz.internal_energy(two_component_identical_dpd)

    assert np.allclose(U1, U1_pair)
    assert np.allclose(U1, U2, atol=1e-4)
    assert np.allclose(U1, U2_inf_pair)
    assert np.allclose(U1, U2_inf)
