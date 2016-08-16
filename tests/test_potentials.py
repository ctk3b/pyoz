import numpy as np
import simtk.unit as u
from simtk.unit import AVOGADRO_CONSTANT_NA as Na
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB
import pytest

from pyoz.potentials import ContinuousPotential, LennardJones


@pytest.mark.skipif(True, reason='Not yet implemented')
def test_custom_function():
    r = np.arange(100)
    T = 300 * u.kelvin
    U = ContinuousPotential('4 * e * ((s / r)**12 - (s / r)**6)',
                            s='arithmetic',
                            e='geometric')
    U.add_parameters('foo',
                     sig=0.5,
                     eps=0.1 * u.kilojoules_per_mole / Na / kB / T)

    U.add_parameters('bar',
                     sig=0.4,
                     eps=0.2 * u.kilojoules_per_mole / Na / kB / T)
    U.apply(r, T)

    U_lj = LennardJones()
    U_lj.add_parameters('foo',
                        sig=0.5 * u.angstroms,
                        eps=0.1 * u.kilojoules_per_mole)
    U_lj.add_parameters('bar',
                        sig=0.4 * u.angstroms,
                        eps=0.2 * u.kilojoules_per_mole)
    U_lj.apply(r, T)

    assert np.allclose(U_lj.ij, U.ij)


if __name__ == '__main__':
    test_custom_function()
