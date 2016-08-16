import numpy as np


def hypernetted_chain(U, G_r):
    """Apply the hyper-etted chains closure.

    g_r = exp(-U) * exp(G_r)
    h_r = g_r - 1
    c_r = exp(-U) * exp(G_r) - G_r - 1

    """
    g_r = np.exp(-U.ij) * np.exp(U.erf_real) * np.exp(G_r)
    c_r = g_r - G_r - 1
    return c_r, g_r


def percus_yevick(U, G_r):
    """Apply the Percus-Yevick closure.

    g_r = exp(-U) * (1 + G_r)
    h_r = g_r - 1
    c_r = exp(-U) * (1 + G_r) - G_r - 1

    """
    g_r = np.exp(-U.ij) * np.exp(U.erf_real) * (1 + G_r)
    c_r = g_r - G_r - 1
    return c_r, g_r


supported_closures = {'hnc': hypernetted_chain,
                      'py': percus_yevick}
