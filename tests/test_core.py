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

    for _ in range(n_potentials):
        comp.add_potential(oz.LennardJones(), sig=1, eps=0.8)
    assert comp.n_potentials == n_potentials


def test_add_pot_vs_add_parm():
    p1 = oz.LennardJones()
    p2 = oz.LennardJones()
    c1 = oz.Component('1')
    c2 = oz.Component('2')

    p1.add_component(c1, sig=5, eps=10)
    c2.add_potential(p2, sig=5, eps=10)

    assert all(p1.parameters.iloc[0] == p2.parameters.iloc[0])
    assert all(c1.parameters[p1].values == c2.parameters[p2].values)


def test_add_component():
    p1 = oz.LennardJones()
    c1 = oz.Component('1')

    p1.add_component(c1, sig=5, eps=10)
    with pytest.raises(PyozError):
        p1.add_component(c1, sig=5, eps=10, z=15)
        p1.add_component(c1)


def test_remove_component():
    p1 = oz.LennardJones()
    c1 = oz.Component('1')
    c2 = oz.Component('2')
    p1.add_component(c1, sig=5, eps=10)
    p1.add_component(c2, sig=50, eps=100)

    p1.remove_component(c2)
    assert p1.n_components == 1
    assert p1.parameters.iloc[0]['sig'] == 5
    assert p1.parameters.iloc[0]['eps'] == 10
    assert p1.parm_ij[0, 0, 0] == 10
    assert p1.parm_ij[1, 0, 0] == 5

    p1 = oz.LennardJones()
    c1 = oz.Component('1')
    c2 = oz.Component('2')
    p1.add_component(c1, sig=5, eps=10)
    p1.add_component(c2, sig=50, eps=100)

    p1.remove_component(c1)
    assert p1.n_components == 1
    assert p1.parameters.iloc[0]['sig'] == 50
    assert p1.parameters.iloc[0]['eps'] == 100
    assert p1.parm_ij[0, 0, 0] == 100
    assert p1.parm_ij[1, 0, 0] == 50


def test_remove_potential():
    p1 = oz.LennardJones()
    c1 = oz.Component('1')
    p1.add_component(c1, sig=5, eps=10)

    c1.remove_potential(p1)
    assert p1.n_components == 0


def test_replace_potential():
    p1 = oz.LennardJones()
    p2 = oz.LennardJones()
    c1 = oz.Component('1')
    p1.add_component(c1, sig=5, eps=10)

    c1.replace_potential(p1, p2, sig=50, eps=100)
    assert p1.n_components == 0
    assert p2.n_components == 1
    assert p2.parameters.iloc[0]['sig'] == 50
    assert p2.parameters.iloc[0]['eps'] == 100
    assert p2.parm_ij[0, 0, 0] == 100
    assert p2.parm_ij[1, 0, 0] == 50


def test_start_solve():
    s1 = oz.System(T=1)
    p1 = oz.LennardJones()
    c1 = oz.Component('1')
    p1.add_component(c1, sig=1, eps=1)

    with pytest.raises(PyozError):
        s1.solve(closure_name='hnc')

    s1.add_component(c1)

    with pytest.raises(PyozError):
        s1.solve(closure_name='foobar')

    s1.solve(closure_name='hnc')
