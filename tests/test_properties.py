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


from scipy.integrate import simps as integrate


def integrate_dat(one):
    r, g_r, U_r = one.r, one.g_r, one.U_r
    dr = r[1] - r[0]
    dUdr = (np.diff(U_r) / dr)
    return integrate(y=r[1:]**3 * g_r[:, :, 1:] * dUdr, x=r[1:])
    # return integrate(y=r[1:]**3 * g_r[:, :, 1:], x=r[1:])
    # return integrate(y=g_r[:, :, 1:], x=r[1:])


def test_debug(one_component_lj,
               two_component_one_inf_dilute_lj):

    one = integrate_dat(one_component_lj)
    two = integrate_dat(two_component_one_inf_dilute_lj)
    assert np.allclose(one, two[0, 0])

def test_debug1(one_component_lj,
               two_component_one_inf_dilute_lj):

    one = one_component_lj
    two = two_component_one_inf_dilute_lj
    assert np.allclose(one.g_r[0, 0], two.g_r[0, 0])
    assert np.allclose(one.g_r[0, 0], two.g_r[0, 0])

def test_debug2(one_component_lj,
               two_component_one_inf_dilute_lj):

    one = one_component_lj
    two = two_component_one_inf_dilute_lj

    dr = one.r[1] - one.r[0]
    du_1 = np.diff(one.U_r, dr)
    du_2 = np.diff(two.U_r, dr)
    assert np.allclose(du_1[0, 0], du_2[0, 0])
