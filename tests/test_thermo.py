import numpy as np
import pytest

import pyoz as oz


# TODO: dummy system as py.test fixture

@pytest.mark.skipif(True, reason='Not yet implemented')
def test_kirkwood_buff():
    dr = 0.05
    n_points = 4095
    r = np.linspace(dr, n_points * dr - dr, n_points)
    g_r = r  # TODO: generate
    oz.kirkwood_buff_integrals()
