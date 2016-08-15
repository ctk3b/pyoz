from hypothesis import given, assume
from hypothesis.strategies import floats, integers, text
import simtk.unit as u

import pyoz as oz


@given(name=text(), dr=floats(), n_points=integers())
def test_init_system(name, dr, n_points):
    assume(dr > 0)
    assume(0 < n_points < 8192)
    oz.System(name=name, dr=dr * u.nanometers, n_points=n_points)


@given(name=text(), concentration=floats())
def test_init_component(name, concentration):
    assume(concentration >= 0)
    oz.Component(name=name, concentration=concentration * u.moles / u.liters)


@given(n_potentials=integers())
def test_add_potential(n_potentials):
    assume(0 < n_potentials < 10)
    comp = oz.Component(name='foo', concentration=0.5 * u.moles / u.liters)
    for _ in range(n_potentials):
        comp.add_potential(oz.LennardJones(),
                           parameters={'sig': 0.1 * u.nanometers,
                                       'eps': 0.2 * u.kilojoules_per_mole})
    assert comp.n_potentials == n_potentials
