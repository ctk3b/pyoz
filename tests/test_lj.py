import itertools as it

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import simtk.unit as u

import pyoz as oz


@given(st.floats(min_value=250, max_value=400),
       st.tuples(st.floats(min_value=0.1, max_value=0.5),
                 st.floats(min_value=0.1, max_value=0.5)),
       st.tuples(st.floats(min_value=0.1, max_value=1),
                 st.floats(min_value=0.1, max_value=1)),
       st.tuples(st.floats(min_value=0.1, max_value=1),
                 st.floats(min_value=0.1, max_value=1)))
def test_two_comp_picard(T, C, sig, eps):
    inputs = dict()
    # Algorithm control.
    inputs['n_points'] = 4096  # use power of 2!
    inputs['dr'] = 0.05 * u.angstrom
    inputs['mix_param'] = 1.0
    inputs['tol'] = 1e-9
    inputs['max_iter'] = 5000

    # System information.
    inputs['T'] = T * u.kelvin
    n_components = 2
    inputs['n_components'] = n_components
    inputs['closure'] = 'hnc'
    inputs['names'] = ['P', 'M']
    inputs['concentrations'] = [C[0] * u.moles / u.liter,
                                C[1] * u.moles / u.liter]

    # Potential parameters.
    inputs['potentials'] = dict()

    lj = inputs['potentials']['lennard-jones'] = dict()
    lj['sigmas'] = [sig[0] * u.nanometers,
                    sig[1] * u.nanometers]
    lj['sigma_rule'] = 'arithmetic'
    lj['epsilons'] = [eps[0] * u.kilojoules_per_mole,
                      eps[1] * u.kilojoules_per_mole]
    lj['epsilon_rule'] = 'geometric'

    r, g_r = oz.solve_ornstein_zernike(inputs)
    assert np.allclose(g_r[:, :, 0],
                       np.zeros(shape=(n_components, n_components)))
    assert np.allclose(g_r[:, :, -1],
                       np.ones(shape=(n_components, n_components)))

if __name__ == '__main__':
    test_two_comp_picard(T=298.15,
                         C=(0.1, 0.1),
                         sig=(0.4, 0.6),
                         eps=(0.1, 0.2))
