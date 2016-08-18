import numpy as np
import pytest

from pyoz.potentials import ContinuousPotential, LennardJones, _LennardJones
import pyoz.unit as u
from pyoz.unit import AVOGADRO_CONSTANT_NA as Na
from pyoz.unit import BOLTZMANN_CONSTANT_kB as kB


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_custom_function():
    r = np.arange(100)
    T = 300 * u.kelvin
    sig = [0.5, 0.4, 0.3]
    eps = [0.1, 0.2, 0.7]
    no_unit = u.kilojoules_per_mole / Na / kB / T
    kj_mol = u.kilojoules_per_mole

    def lj_func(r, e, s):
        return 4 * e * ((s / r)**12 - (s / r)**6)

    # Generic ContinuousPotential.
    U_cont = ContinuousPotential(lj_func, s='arithmetic', e='geometric')
    U_cont.add_parameters('foo', s=sig[0], e=eps[0] * no_unit)
    U_cont.add_parameters('bar', s=sig[1], e=eps[1] * no_unit)
    U_cont.add_parameters('baz', s=sig[2], e=eps[2] * no_unit)
    U_cont.apply(r, T)

    # Subclassed ContinuousPotential.
    U_sub = LennardJones(s='arithmetic', e='geometric')
    U_sub.add_parameters('foo', s=sig[0], e=eps[0] * no_unit)
    U_sub.add_parameters('bar', s=sig[1], e=eps[1] * no_unit)
    U_sub.add_parameters('baz', s=sig[2], e=eps[2] * no_unit)
    U_sub.apply(r, T)

    # Hardcoded LennardJones, primarily for testing purposes.
    U_lj = _LennardJones()
    U_lj.add_parameters('foo', sig=sig[0] * u.angstrom, eps=eps[0] * kj_mol)
    U_lj.add_parameters('bar', sig=sig[1] * u.angstrom, eps=eps[1] * kj_mol)
    U_lj.add_parameters('baz', sig=sig[2] * u.angstrom, eps=eps[2] * kj_mol)
    U_lj.apply(r, T)

    assert np.allclose(U_lj.ij, U_cont.ij, equal_nan=True)
    assert np.allclose(U_lj.ij, U_sub.ij, equal_nan=True)


if __name__ == '__main__':
    test_custom_function()
