import numpy as np
import simtk.unit as u
from simtk.unit import AVOGADRO_CONSTANT_NA as Na
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB
import pytest

from pyoz.potentials import ContinuousPotential, LennardJones, _LennardJones


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_custom_function():
    r = np.arange(100)
    T = 300 * u.kelvin

    def lj_func(r, e, s):
        return 4 * e * ((s / r)**12 - (s / r)**6)

    # Generic ContinuousPotential.
    U_cont = ContinuousPotential(lj_func, s='arithmetic', e='geometric')
    U_cont.add_parameters('foo',
                          s=0.5,
                          e=0.1 * u.kilojoules_per_mole / Na / kB / T)
    U_cont.add_parameters('bar',
                          s=0.4,
                          e=0.2 * u.kilojoules_per_mole / Na / kB / T)
    U_cont.add_parameters('baz',
                          s=0.3,
                          e=0.7 * u.kilojoules_per_mole / Na / kB / T)
    U_cont.apply(r, T)

    # Subclassed ContinuousPotential.
    U_sub = LennardJones(s='arithmetic', e='geometric')
    U_sub.add_parameters('foo',
                         s=0.5,
                         e=0.1 * u.kilojoules_per_mole / Na / kB / T)
    U_sub.add_parameters('bar',
                         s=0.4,
                         e=0.2 * u.kilojoules_per_mole / Na / kB / T)
    U_sub.add_parameters('baz',
                         s=0.3,
                         e=0.7 * u.kilojoules_per_mole / Na / kB / T)
    U_sub.apply(r, T)

    # Hardcoded LennardJones, primarily for testing purposes.
    U_lj = _LennardJones()
    U_lj.add_parameters('foo',
                        sig=0.5 * u.angstroms,
                        eps=0.1 * u.kilojoules_per_mole)
    U_lj.add_parameters('bar',
                        sig=0.4 * u.angstroms,
                        eps=0.2 * u.kilojoules_per_mole)

    U_lj.add_parameters('baz',
                        sig=0.3 * u.angstroms,
                        eps=0.7 * u.kilojoules_per_mole)
    U_lj.apply(r, T)

    assert np.allclose(U_lj.ij, U_cont.ij, equal_nan=True)
    assert np.allclose(U_lj.ij, U_sub.ij, equal_nan=True)


if __name__ == '__main__':
    test_custom_function()
