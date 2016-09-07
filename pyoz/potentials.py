import numpy as np

from pyoz.exceptions import PyozError

__all__ = ['arithmetic', 'geometric', 'mie', 'lennard_jones', 'wca', 'coulomb',
           'screened_coulomb']

def arithmetic(a, b):
    return 0.5 * (a + b)


def geometric(a, b):
    return np.sqrt(a * b)


def mie(r, eps, sig, m, n):
    return 4 * eps * ((sig / r)**m - (sig / r)**n)


def lennard_jones(r, eps, sig):
    return mie(r, eps, sig, m=12, n=6)


def wca(r, eps, sig, m, n):
    p = 1 / (m - n)
    r_cut = sig * (m / n)**p
    U = lennard_jones(r, eps, sig) + eps
    return np.where(r < r_cut, U, 0)


def coulomb(r, q1, q2, bjerrum_length=1):
    return bjerrum_length * q1 * q2 / r


def screened_coulomb(r, q1, q2, bjerrum_length=1, debye_length=1):
    return coulomb(r, q1, q2, bjerrum_length) * np.exp(-r / debye_length)
