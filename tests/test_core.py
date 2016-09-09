from math import isclose

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

    s.set_interaction(0, 0, range(s.n_points))
    assert s.U_r.shape == (1, 1, s.n_points)

    s.set_interaction(1, 1, range(s.n_points))
    assert s.U_r.shape == (2, 2, s.n_points)

    s.set_interaction(0, 1, range(s.n_points))
    assert s.U_r.shape == (2, 2, s.n_points)
    assert (s.U_r[0, 1] == s.U_r[1, 0]).all()

    s.set_interaction(0, 4, range(s.n_points))
    assert s.U_r.shape == (5, 5, s.n_points)

    s.set_interaction(0, 5, range(s.n_points), symmetric=False)
    assert s.U_r.shape == (6, 6, s.n_points)
    assert (s.U_r[0, 5] != s.U_r[5, 0]).any()
    assert (s.U_r[5, 0] == 0).all()


@pytest.mark.skipif(True, reason='Not implemented yet')
def test_remove_interaction():
    s = oz.System()

    s.set_interaction(0, 0, range(s.n_points))
    s.set_interaction(1, 1, range(10, 10 + s.n_points))
    s.set_interaction(2, 2, range(20, 20 + s.n_points))
    s.set_interaction(3, 3, range(30, 30 + s.n_points))

    s.remove_interaction(0, 1)
    assert s.U_r.shape == (4, 4, s.n_points)

    s.remove_interaction(3, 3)
    assert s.U_r.shape == (3, 3, s.n_points)
    assert s.U_r[0, 0, -1] == s.n_points
    assert s.U_r[1, 1, -1] == s.n_points + 10
    assert s.U_r[2, 2, -1] == s.n_points + 10

    s.remove_interaction(0, 0)
    assert s.U_r.shape == (2, 2, s.n_points)
    assert s.U_r[0, 0, -1] == s.n_points + 10
    assert s.U_r[1, 1, -1] == s.n_points + 20


def test_start_solve():
    s1 = oz.System(T=1)

    with pytest.raises(TypeError):
        s1.solve()

    s1.set_interaction(0, 0, oz.wca(s1.r, eps=1, sig=1, m=12, n=6))

    with pytest.raises(PyozError):
        s1.solve(rhos=[0.1], closure_name='foobar')

    s1.solve(rhos=[0.1])
