import numpy as np

from pyoz.exceptions import PyozError
import pyoz
try:
    from numba import jit
except ImportError:
    def jit(f):
        return f
    pyoz.logger.warn('Unable to import `numba`. Installing `numba` will '
                     'significantly accelerate your code:\n\n'
                     '"conda install numba"\n\n')


def rms_normed(A, B):
    """Compute the squared and normed distance between two arrays.

    distance = sqrt(sum[(A - B)^2] / (n_points * n_components^2))
    """
    distance = ((A - B) ** 2).sum()
    distance /= A.shape[2] * A.shape[0] ** 2
    return np.sqrt(distance)


@jit(nopython=True)
def solver(A, B):
    """Solve the matrix problem in fourier space.

    Note that the convolution theorem involves a constant factor ('ff')
    depending on the forward fourier transform normalization constant.

    H = C + ff CH
    H - ff CH = C
    {E - ff C}H = C
    H = {E - ff C}^-1 * C
    h = H / dens_factor
    """
    n_components = A.shape[0]
    n_points = A.shape[-1]

    if n_components == 1:
        if (A == 0).any():
            raise PyozError('Singular matrix, cannot invert')
        H_k = B / A
    elif n_components == 2:
        H_k = np.empty_like(A)
        A_det = A[0, 0]*A[1, 1] - A[1, 0]*A[0, 1]
        if (A_det == 0.0).any():
            raise PyozError('Singular matrix, cannot invert')

        for dr in range(n_points):
            A_inv = np.ones(shape=(2, 2,)) / A_det[dr]
            A_inv[0, 0] *= A[1, 1, dr]
            A_inv[0, 1] *= -A[0, 1, dr]
            A_inv[1, 0] *= -A[1, 0, dr]
            A_inv[1, 1] *= A[0, 0, dr]

            # h[:,:,dr] = (mat(a_inv[dr]) * mat(b[:,:,dr])) / dens_factor
            # explicitly - is faster
            H_k[0, 0, dr] = A_inv[0, 0]*B[0, 0, dr] + A_inv[0, 1]*B[1, 0, dr]
            H_k[0, 1, dr] = A_inv[0, 0]*B[0, 1, dr] + A_inv[0, 1]*B[1, 1, dr]
            H_k[1, 0, dr] = A_inv[1, 0]*B[0, 0, dr] + A_inv[1, 1]*B[1, 0, dr]
            H_k[1, 1, dr] = A_inv[1, 0]*B[0, 1, dr] + A_inv[1, 1]*B[1, 1, dr]
    elif A.shape[0] >= 2:
        H_k = np.empty_like(A)
        for dr in range(n_points):
            H_k[:, :, dr] = np.linalg.solve(A[:, :, dr], B[:, :, dr])
    return H_k


