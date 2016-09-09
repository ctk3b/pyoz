import numpy as np

import pyoz as oz


def test_two_comp_picard():
    T = 1
    sig = np.array([1, 2])
    eps = np.array([1, 0.75])
    rhos = np.array([0.01, 0.01]) / sig**3

    oz.logger.info('T=  {:8.2f}'.format(T))
    oz.logger.info('C=  {:8.2f}{:8.2f}'.format(*rhos))
    oz.logger.info('sig={:8.2f}{:8.2f}'.format(*sig))
    oz.logger.info('eps={:8.2f}{:8.2f}'.format(*eps))

    lj_liquid = oz.System(T=T)

    r = lj_liquid.r
    lj_liquid.set_interaction(0, 0, oz.lennard_jones(r, eps[0] / T, sig[0]))
    lj_liquid.set_interaction(1, 1, oz.lennard_jones(r, eps[1] / T, sig[1]))

    eps_01 = oz.geometric(eps[0], eps[1])
    sig_01 = oz.arithmetic(sig[0], sig[1])
    lj_liquid.set_interaction(0, 1, oz.lennard_jones(r, eps_01 / T, sig_01))

    lj_liquid.solve(rhos=rhos, closure_name='hnc')

    n_components = lj_liquid.n_components
    assert np.allclose(lj_liquid.g_r[:, :, :10],
                       np.zeros(shape=(n_components, n_components, 10)))
    assert np.allclose(lj_liquid.g_r[:, :, -10:],
                       np.ones(shape=(n_components, n_components, 10)))
