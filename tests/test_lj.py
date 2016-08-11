from copy import deepcopy

from hypothesis import given
from hypothesis.strategies import floats, tuples
import numpy as np
import simtk.unit as u

import pyoz as oz


@given(T=floats(min_value=250, max_value=400),
       C=tuples(floats(min_value=0.1, max_value=0.5),
                floats(min_value=0.1, max_value=0.5)),
       sig=tuples(floats(min_value=0.1, max_value=1),
                  floats(min_value=0.1, max_value=1)),
       eps=tuples(floats(min_value=0.1, max_value=1),
                  floats(min_value=0.1, max_value=1)))
def test_two_comp_picard(T, C, sig, eps):
    oz.logger.info('T=  {:8.2f}'.format(T))
    oz.logger.info('C=  {:8.2f}{:8.2f}'.format(*C))
    oz.logger.info('sig={:8.2f}{:8.2f}'.format(*sig))
    oz.logger.info('eps={:8.2f}{:8.2f}'.format(*eps))
    inputs = deepcopy(oz.settings)
    inputs['T'] = T * u.kelvin
    n_components = 2
    inputs['n_components'] = n_components
    inputs['concentrations'] = [C[0] * u.moles / u.liter,
                                C[1] * u.moles / u.liter]

    # Potential parameters.
    lj = inputs['potentials']['lennard-jones']
    lj['sigmas'] = [sig[0] * u.nanometers,
                    sig[1] * u.nanometers]
    lj['sigma_rule'] = 'arithmetic'
    lj['epsilons'] = [eps[0] * u.kilojoules_per_mole,
                      eps[1] * u.kilojoules_per_mole]
    lj['epsilon_rule'] = 'geometric'

    r, g_r = oz.solve_ornstein_zernike(inputs, status_updates=False)
    assert np.allclose(g_r[:, :, 0],
                       np.zeros(shape=(n_components, n_components)))
    assert np.allclose(g_r[:, :, -1],
                       np.ones(shape=(n_components, n_components)))

if __name__ == '__main__':
    test_two_comp_picard(T=298.15,
                         C=(0.1, 0.1),
                         sig=(0.4, 0.6),
                         eps=(0.1, 0.2))
