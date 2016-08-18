import numpy as np
import scipy.integrate as integrate


def rms_normed(A, B):
    """Compute the squared and normed distance between two arrays.

    distance = sqrt(sum[(A - B)^2] / (n_points * n_components^2))
    """
    distance = ((A - B) ** 2).sum()
    distance /= A.shape[2] * A.shape[0] ** 2
    return np.sqrt(distance)


def find_nearest(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]


def dotproduct(ctrl, syst, r, X_ij, Y_ij):
    """calculates the dot product of two functions X, Y discretized on N points and represented
       as square matrices at every discretization point according to

       dotprod = sum_ij [rho_i rho_j \int X_ij(r) Y_ij(r) 4 \pi r^2 dr]

       with rho being constant factors stored in a square matrix of the same dimension as X(r) and Y(r)
    """
    dotprod = 0.0

    # calculate the integrand (array product)
    integrand = 4.0 * np.pi * X_ij * Y_ij

    for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
            integrand[i, j, :] *= syst['dens']['num'][i] * syst['dens']['num'][j] * r ** 2
            dotprod += integrate.simps(integrand[i, j], r,
                                       dx=ctrl['deltar'], even='last')
    return dotprod


def interpolate_linear(val1, val2, mu):
    """linear interpolation

      val1 and val2 are values at positions x1, x2
      target point is specified by mu
      mu=0 => x1, mu=1 => x2
    """
    return val1 * (1.0 - mu) + val2 * mu


def interpolate_cosine(val1, val2, mu):
    """Cosine interpolation

      val1 and val2 are values at positions x1, x2
      target point is specified by mu
      mu=0 => x1, mu=1 => x2
    """
    mu2 = (1.0 - np.cos(mu * np.pi)) / 2.0
    return val1 * (1.0 - mu2) + val2 * mu2



