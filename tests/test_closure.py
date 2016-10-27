import numpy as np

import pyoz as oz


def test_compare_hnc_kh():
    rho = 5

    a = oz.System()
    b = oz.System()
    U = oz.dpd(a.r, 10)


    a.set_interaction(0, 0, U)
    hnc = a.solve(rhos=rho, closure_name='hnc')

    b.set_interaction(0, 0, U)
    kh = b.solve(rhos=rho, closure_name='kh')

    for a_x, b_x in zip(hnc, kh):
        assert np.allclose(a_x, b_x, atol=1e-2, rtol=1e-2)


def test_compare_hnc_rhnc():
    rho = 0.0001

    a = oz.System()
    b = oz.System()
    b_ref = oz.System()

    U = oz.lennard_jones(a.r, eps=1, sig=1)
    U_ref = oz.wca(a.r, eps=1, sig=1, m=12, n=6)


    a.set_interaction(0, 0, U)
    hnc = a.solve(rhos=rho, closure_name='hnc')

    b_ref.set_interaction(0, 0, U_ref)
    b.set_interaction(0, 0, U)
    rhnc = b.solve(rhos=rho, closure_name='rhnc', reference_system=b_ref)

    for a_x, b_x in zip(hnc, rhnc):
        assert np.allclose(a_x, b_x, atol=1e-3, rtol=1e-3)

