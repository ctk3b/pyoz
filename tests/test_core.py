from math import isclose

import numpy as np
import pytest

import pyoz as oz
from pyoz.exceptions import PyozError


def test_init_system():
    name = 'A'
    dr = 0.05
    n_points = 4096
    s = oz.System(name=name, dr=dr, n_points=n_points)
    assert len(s.r) == n_points - 1
    assert isclose(s.r[1] - s.r[0], dr, rel_tol=1e-3)


def test_add_interaction():
    s = oz.System()

    with pytest.raises(PyozError):
        s.set_interaction(0, 0, range(s.n_pts - 1))

    s.set_interaction(0, 0, range(s.n_pts))
    assert s.U_r.shape == (1, 1, s.n_pts)

    s.set_interaction(1, 1, range(s.n_pts))
    assert s.U_r.shape == (2, 2, s.n_pts)

    s.set_interaction(0, 1, range(s.n_pts))
    assert s.U_r.shape == (2, 2, s.n_pts)
    assert (s.U_r[0, 1] == s.U_r[1, 0]).all()

    s.set_interaction(0, 4, range(s.n_pts))
    assert s.U_r.shape == (5, 5, s.n_pts)


@pytest.mark.skipif(True, reason='Not implemented yet')
def test_remove_interaction():
    s = oz.System()

    s.set_interaction(0, 0, range(s.n_pts))
    s.set_interaction(1, 1, range(10, 10 + s.n_pts))
    s.set_interaction(2, 2, range(20, 20 + s.n_pts))
    s.set_interaction(3, 3, range(30, 30 + s.n_pts))

    s.remove_interaction(0, 1)
    assert s.U_r.shape == (4, 4, s.n_pts)

    s.remove_interaction(3, 3)
    assert s.U_r.shape == (3, 3, s.n_pts)
    assert s.U_r[0, 0, -1] == s.n_pts
    assert s.U_r[1, 1, -1] == s.n_pts + 10
    assert s.U_r[2, 2, -1] == s.n_pts + 20

    s.remove_interaction(0, 0)
    assert s.U_r.shape == (2, 2, s.n_pts)
    assert s.U_r[0, 0, -1] == s.n_pts + 10
    assert s.U_r[1, 1, -1] == s.n_pts + 20


def test_resolve():
    s1 = oz.System()

    s1.set_interaction(0, 0, oz.wca(s1.r, eps=1, sig=1, m=12, n=6))
    results_1 = s1.solve(rhos=[0.1])

    s1.set_interaction(0, 0, oz.wca(s1.r, eps=2, sig=2, m=18, n=12))
    results_2 = s1.solve(rhos=[0.1])

    for array_1, array_2 in zip(results_1, results_2):
        assert not np.allclose(array_1, array_2)


def test_start_solve():
    s1 = oz.System()

    with pytest.raises(TypeError):
        s1.solve()

    with pytest.raises(PyozError):
        s1.solve(rhos=[0.1])

    s1.set_interaction(0, 0, oz.wca(s1.r, eps=1, sig=1, m=12, n=6))

    with pytest.raises(PyozError):
        s1.solve(rhos=[0.1], closure_name='foobar')

    with pytest.raises(PyozError):
        s1.solve(rhos=[0.1, 0.2], closure_name='HNC')

    s1.solve(rhos=[0.1])


def test_two_component_identical(one_component_dpd, two_component_identical_dpd):
    one = one_component_dpd
    two = two_component_identical_dpd

    assert np.allclose(one.U_r[0, 0], two.U_r[0, 0], equal_nan=True)
    assert np.allclose(one.U_r[0, 0], two.U_r[0, 1], equal_nan=True)
    assert np.allclose(one.U_r[0, 0], two.U_r[1, 1], equal_nan=True)

    assert np.allclose(one.g_r[0, 0], two.g_r[0, 0], equal_nan=True)
    assert np.allclose(one.g_r[0, 0], two.g_r[0, 1], equal_nan=True)
    assert np.allclose(one.g_r[0, 0], two.g_r[1, 1], equal_nan=True)

    assert np.allclose(one.h_k[0, 0] / 2, two.h_k[0, 0], equal_nan=True)
    assert np.allclose(one.h_k[0, 0] / 2, two.h_k[0, 1], equal_nan=True)
    assert np.allclose(one.h_k[0, 0] / 2, two.h_k[1, 1], equal_nan=True)


def test_one_inf_dilute(one_component_dpd, two_component_one_inf_dilute_dpd):
    one = one_component_dpd
    two = two_component_one_inf_dilute_dpd

    assert np.allclose(one.U_r[0, 0], two.U_r[0, 0])
    assert np.allclose(one.g_r[0, 0], two.g_r[0, 0])
    assert np.allclose(one.h_k[0, 0], two.h_k[0, 0])

    assert np.allclose(two.g_r[0, 1], np.exp(-two.U_r[0, 1]))
    assert np.allclose(two.g_r[1, 1], np.exp(-two.U_r[1, 1]))
    assert not np.any(two.h_k[0, 1])
    assert not np.any(two.h_k[1, 1])


def test_solve_with_reference():
    eps = 1
    sig = 1
    wca_ref = oz.System()
    wca_ref.set_interaction(0, 0, oz.wca(wca_ref.r, eps=eps, sig=sig, m=12, n=6))

    lj = oz.System()
    lj.set_interaction(0, 0, oz.lennard_jones(lj.r, eps=eps, sig=sig))

    with pytest.raises(PyozError):
        lj.solve(rhos=0.01, closure_name='RHNC')
    lj.solve(rhos=0.01, closure_name='RHNC', reference_system=wca_ref)


def test_unconverged():
    lj = oz.System()

    r = lj.r
    lj.set_interaction(0, 0, oz.lennard_jones(r, 1, 1))

    g_r, c_r, e_r, h_k = lj.solve(rhos=[10], closure_name='hnc')
    assert np.isnan(g_r).all()
    assert np.isnan(c_r).all()
    assert np.isnan(e_r).all()
    assert np.isnan(h_k).all()

