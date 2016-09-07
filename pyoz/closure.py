import numpy as np


def hypernetted_chain(U_r, G_r, U_r_erf_real, **kwargs):
    """Apply the hyper-netted chains closure.

    g_r = exp(-U) * exp(G_r)
    h_r = g_r - 1
    c_r = exp(-U) * exp(G_r) - G_r - 1

    """
    g_r = np.exp(-U_r) * np.exp(U_r_erf_real) * np.exp(G_r)
    c_r = g_r - G_r - 1
    return c_r, g_r


def reference_hypernetted_chain(U_r, G_r, U_r_erf_real, **kwargs):
    """Apply the hyper-netted chains closure.

    g_r = exp(-U) * exp(G_r)
    h_r = g_r - 1
    c_r = exp(-U) * exp(G_r) - G_r - 1

    """
    g_r_ref, G_r_ref, U_r_ref = kwargs['g_r_ref'], kwargs['G_r_ref'], kwargs['U_r_ref']
    dU = U_r - U_r_ref
    dG = G_r - G_r_ref
    c_r = g_r_ref * np.exp(-dU + dG) - G_r - 1
    g_r = c_r + G_r + 1
    return c_r, g_r


def percus_yevick(U_r, G_r, U_r_erf_real, **kwargs):
    """Apply the Percus-Yevick closure.

    g_r = exp(-U) * (1 + G_r)
    h_r = g_r - 1
    c_r = exp(-U) * (1 + G_r) - G_r - 1

    """
    g_r = np.exp(-U_r) * np.exp(U_r_erf_real) * (1 + G_r)
    c_r = g_r - G_r - 1
    return c_r, g_r


supported_closures = {'HNC': hypernetted_chain,
                      'RHNC': reference_hypernetted_chain,
                      'PY': percus_yevick}
