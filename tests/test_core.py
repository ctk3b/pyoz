from hypothesis import given, assume
from hypothesis.strategies import floats, integers, text
import pytest

import pyoz as oz
from pyoz.exceptions import PyozError


@given(name=text(), dr=floats(), n_points=integers())
def test_init_system(name, dr, n_points):
    assume(dr > 0)
    assume(1 < n_points < 8192)
    oz.System(name=name, dr=dr, n_points=n_points)


@given(name=text(), concentration=floats())
def test_init_component(name, concentration):
    assume(concentration >= 0)
    oz.Component(name=name, rho=concentration)


@given(n_potentials=integers())
def test_add_potential(n_potentials):
    assume(0 < n_potentials < 10)
    comp = oz.Component(name='foo')
    syst = oz.System()

    for _ in range(n_potentials):
        comp.add_potential(oz.LennardJones(system=syst), sig=1, eps=0.8)
    assert comp.n_potentials == n_potentials


def test_add_pot_vs_add_parm():
    syst = oz.System()

    def lj_func(r, e, s):
        return 4 * e * ((s / r)**12 - (s / r)**6)
    p1 = oz.ContinuousPotential(system=syst, potential_func=lj_func)
    p2 = oz.ContinuousPotential(system=syst, potential_func=lj_func)
    c1 = oz.Component('1')
    c2 = oz.Component('2')

    p1.add_component(c1, s=5, e=10)
    c2.add_potential(p2, s=5, e=10)

    assert all(p1.parameters.iloc[0] == p2.parameters.iloc[0])
    assert all(c1.parameters[p1].values == c2.parameters[p2].values)


def test_add_component():
    syst = oz.System()

    def lj_func(r, e, s):
        return 4 * e * ((s / r)**12 - (s / r)**6)
    p1 = oz.ContinuousPotential(system=syst, potential_func=lj_func)
    c1 = oz.Component('1')

    p1.add_component(c1, s=5, e=10)
    with pytest.raises(PyozError):
        p1.add_component(c1, s=5, e=10, z=15)
        p1.add_component(c1)
