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

    lj_liquid = oz.System(T=T * u.kelvin)
    potential = oz.LennardJones(sig_rule='arithmetic', eps_rule='geometric')

    m = oz.Component(name='M', concentration=C[0] * u.moles / u.liter)
    m.add_potential(potential, parameters={'sig': sig[0] * u.nanometers,
                                           'eps': eps[0] * u.kilojoules_per_mole})
    n = oz.Component(name='N', concentration=C[1] * u.moles / u.liter)
    n.add_potential(potential, parameters={'sig': sig[0] * u.nanometers,
                                           'eps': eps[0] * u.kilojoules_per_mole})

    lj_liquid.add_component(m)
    lj_liquid.add_component(n)

    r, g_r = lj_liquid.solve(closure='hnc')

    n_components = lj_liquid.n_components
    assert np.allclose(g_r[:, :, :10],
                       np.zeros(shape=(n_components, n_components, 10)))
    assert np.allclose(g_r[:, :, -10:],
                       np.ones(shape=(n_components, n_components, 10)))

if __name__ == '__main__':
    test_two_comp_picard(T=298.15,
                         C=(0.1, 0.1),
                         sig=(0.4, 0.6),
                         eps=(0.1, 0.2))
