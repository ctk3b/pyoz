import numpy as np

import pyoz as oz


def test_mie():
    r = np.arange(1, 101)
    eps = 1
    sig = 1
    assert np.allclose(oz.mie(r, eps, sig, 12, 6),
                       oz.lennard_jones(r, eps, sig))