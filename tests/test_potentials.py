from hypothesis import given
from hypothesis.strategies import floats, tuples
import numpy as np
import pytest

import pyoz as oz
from pyoz.exceptions import PyozError
from pyoz.potentials import _LennardJones
import pyoz.unit as u


@given(sig=tuples(floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000)),
       eps=tuples(floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000)))
def test_custom_function(sig, eps):
    kJ_mol = u.kilojoules_per_mole
    J_mol = u.joules / u.mole
    kcal_mol = u.kilocalories_per_mole

    c1 = oz.Component('1')
    c2 = oz.Component('2')
    c3 = oz.Component('3')

    lj = oz.System()

    # Generic ContinuousPotential.
    def lj_func(r, eps, sig):
        return 4 * eps * ((sig / r)**12 - (sig / r)**6)

    U_cont = oz.ContinuousPotential(system=lj, potential_func=lj_func,
                                    sig='arithmetic', eps='geometric')
    U_cont.add_component(c1, sig=0.1 * sig[0] * u.nanometers, eps=eps[0] * kcal_mol)
    U_cont.add_component(c2, sig=0.1 * sig[1] * u.nanometers, eps=eps[1] * kcal_mol)
    U_cont.add_component(c3, sig=0.1 * sig[2] * u.nanometers, eps=eps[2] * kcal_mol)
    U_cont.apply()

    # Subclassed ContinuousPotential.
    U_sub = oz.LennardJones(system=lj, sig='arithmetic', eps='geometric')
    U_sub.add_component(c1, sig=100 * sig[0] * u.picometers, eps=4184 * eps[0] * J_mol)
    U_sub.add_component(c2, sig=100 * sig[1] * u.picometers, eps=4184 * eps[1] * J_mol)
    U_sub.add_component(c3, sig=100 * sig[2] * u.picometers, eps=4184 * eps[2] * J_mol)
    U_sub.apply()

    # Hardcoded LennardJones, primarily for testing purposes.
    U_lj = _LennardJones(system=lj)
    U_lj.add_component(c1, sig=sig[0] * u.angstrom, eps=4.184 * eps[0] * kJ_mol)
    U_lj.add_component(c2, sig=sig[1] * u.angstrom, eps=4.184 * eps[1] * kJ_mol)
    U_lj.add_component(c3, sig=sig[2] * u.angstrom, eps=4.184 * eps[2] * kJ_mol)
    U_lj.apply()

    assert np.allclose(U_lj.ij, U_cont.ij, equal_nan=True)
    assert np.allclose(U_lj.ij, U_sub.ij, equal_nan=True)


def test_add_binary_interaction():
    lj = oz.System()
    c0 = oz.Component('0')
    c1 = oz.Component('1')
    pot = oz.LennardJones(system=lj)

    parm00 = {'eps': 0, 'sig': 0}
    parm11 = {'eps': 1, 'sig': 1}
    parm01 = {'eps': 0, 'sig': 1}

    with pytest.raises(PyozError):
        pot.add_binary_interaction(c0, c1, **parm01)

    pot.add_component(c0, **parm00)

    with pytest.raises(PyozError):
        pot.add_binary_interaction(c0, c1, **parm01)

    pot.add_component(c1, **parm11)
    pot.add_binary_interaction(c0, c1, **parm01)

    assert pot.parm_ij[0, 0, 0] == pot.parm_ij[1, 0, 0] == 0

    assert pot.parm_ij[0, 1, 1] == pot.parm_ij[1, 1, 1] == 1

    assert pot.parm_ij[0, 0, 1] == pot.parm_ij[0, 1, 0] == 0
    assert pot.parm_ij[1, 0, 1] == pot.parm_ij[1, 1, 0] == 1


def test_mixing_rules():
    lj = oz.System()
    c0 = oz.Component('0')
    c1 = oz.Component('1')
    pot = oz.LennardJones(system=lj, sig='arithmetic', eps='geometric')

    parm00 = {'eps': 0, 'sig': 0}
    parm11 = {'eps': 1, 'sig': 1}

    pot.add_component(c0, **parm00)
    pot.add_component(c1, **parm11)

    assert pot.parm_ij[0, 0, 0] == pot.parm_ij[1, 0, 0] == 0

    assert pot.parm_ij[0, 1, 1] == pot.parm_ij[1, 1, 1] == 1

    assert pot.parm_ij[0, 0, 1] == pot.parm_ij[0, 1, 0] == 0
    assert pot.parm_ij[1, 0, 1] == pot.parm_ij[1, 1, 0] == 0.5


def test_partial_mixing_rules():
    lj = oz.System()
    c0 = oz.Component('0')
    c1 = oz.Component('1')
    pot = oz.LennardJones(system=lj, sig='arithmetic')

    parm00 = {'eps': 1, 'sig': 1}
    parm11 = {'eps': 2, 'sig': 2}

    pot.add_component(c0, **parm00)
    pot.add_component(c1, **parm11)

    assert pot.parm_ij[0, 0, 0] == pot.parm_ij[1, 0, 0] == 1

    assert pot.parm_ij[0, 1, 1] == pot.parm_ij[1, 1, 1] == 2

    assert pot.parm_ij[0, 0, 1] == pot.parm_ij[0, 1, 0] == 0
    assert pot.parm_ij[1, 0, 1] == pot.parm_ij[1, 1, 0] == 1.5


def test_invalid_mixing_rules():
    lj = oz.System()
    with pytest.raises(PyozError):
        pot = oz.LennardJones(system=lj, sig='arithmetic', epz='geometric')
    with pytest.raises(PyozError):
        pot = oz.LennardJones(system=lj,
                              sig='arithmetic',
                              eps='geometric',
                              foo='arithmetic')
