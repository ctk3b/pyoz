from hypothesis import given
from hypothesis.strategies import floats, tuples
import numpy as np

import pyoz as oz


# @given(T=floats(min_value=100, max_value=400),
#        C=tuples(floats(min_value=0.0001, max_value=0.5),
#                 floats(min_value=0.0001, max_value=0.5)),
#        sig=tuples(floats(min_value=1, max_value=10),
#                   floats(min_value=1, max_value=10)),
#        eps=tuples(floats(min_value=50, max_value=150),
#                   floats(min_value=50, max_value=150)))
def test_two_comp_picard():
    T = 298.15
    C = (0.1, 0.1)
    sig = (4.0, 6.0)
    eps = (100, 150)
    oz.logger.info('T=  {:8.2f}'.format(T))
    oz.logger.info('C=  {:8.2f}{:8.2f}'.format(*C))
    oz.logger.info('sig={:8.2f}{:8.2f}'.format(*sig))
    oz.logger.info('eps={:8.2f}{:8.2f}'.format(*eps))

    lj_liquid = oz.System(T=T)
    potential = oz.LennardJones(system=lj_liquid,
                                sig='arithmetic',
                                eps='geometric')

    m = oz.Component(name='M', concentration=C[0] / sig[0]**3)
    m.add_potential(potential, sig=sig[0], eps=eps[0] / T)
    lj_liquid.add_component(m)

    n = oz.Component(name='N', concentration=C[1] / sig[1]**3)
    n.add_potential(potential, sig=sig[1], eps=eps[1] / T)
    lj_liquid.add_component(n)

    lj_liquid.solve(closure='hnc')

    n_components = lj_liquid.n_components
    assert np.allclose(lj_liquid.g_r[:, :, :10],
                       np.zeros(shape=(n_components, n_components, 10)))
    assert np.allclose(lj_liquid.g_r[:, :, -10:],
                       np.ones(shape=(n_components, n_components, 10)))

