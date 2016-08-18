from hypothesis import given
from hypothesis.strategies import floats, tuples
import numpy as np

from pyoz.potentials import ContinuousPotential, LennardJones, _LennardJones
import pyoz.unit as u


@given(sig=tuples(floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000)),
       eps=tuples(floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000),
                  floats(min_value=0.001, max_value=1000)))
def test_custom_function(sig, eps):
    r = np.arange(1, 100)
    T = 300 * u.kelvin
    kJ_mol = u.kilojoules_per_mole
    J_mol = u.joules / u.mole
    kcal_mol = u.kilocalories_per_mole

    # Generic ContinuousPotential.
    def lj_func(r, e, s):
        return 4 * e * ((s / r)**12 - (s / r)**6)

    U_cont = ContinuousPotential(lj_func, s='arithmetic', e='geometric')
    U_cont.add_parameters('foo', s=0.1 * sig[0] * u.nanometers, e=eps[0] * kcal_mol)
    U_cont.add_parameters('bar', s=0.1 * sig[1] * u.nanometers, e=eps[1] * kcal_mol)
    U_cont.add_parameters('baz', s=0.1 * sig[2] * u.nanometers, e=eps[2] * kcal_mol)
    U_cont.apply(r, T)

    # Subclassed ContinuousPotential.
    U_sub = LennardJones(s='arithmetic', e='geometric')
    U_sub.add_parameters('foo', s=100 * sig[0] * u.picometers, e=4184 * eps[0] * J_mol)
    U_sub.add_parameters('bar', s=100 * sig[1] * u.picometers, e=4184 * eps[1] * J_mol)
    U_sub.add_parameters('baz', s=100 * sig[2] * u.picometers, e=4184 * eps[2] * J_mol)
    U_sub.apply(r, T)

    # Hardcoded LennardJones, primarily for testing purposes.
    U_lj = _LennardJones()
    U_lj.add_parameters('foo', sig=sig[0] * u.angstrom, eps=4.184 * eps[0] * kJ_mol)
    U_lj.add_parameters('bar', sig=sig[1] * u.angstrom, eps=4.184 * eps[1] * kJ_mol)
    U_lj.add_parameters('baz', sig=sig[2] * u.angstrom, eps=4.184 * eps[2] * kJ_mol)
    U_lj.apply(r, T)

    assert np.allclose(U_lj.ij, U_cont.ij, equal_nan=True)
    assert np.allclose(U_lj.ij, U_sub.ij, equal_nan=True)
