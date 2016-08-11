import numpy as np


def hypernetted_chain(U, G_r):
    """the HNC closure, calculates the direct correlation function from the pair potential and
       the gamma function; also the total and pair correlation functions

       according to HNC
       g_r = exp(-U) * exp(G_r)
       h_r = g_r - 1
       c_r = exp(-U) * exp(G_r) - G_r - 1

       the discontinuities of the interaction potentials are taken care of here as well
       with help of the U_discontinuity list
    """
    g_r = np.exp(-U.ij) * np.exp(U.erf_real) * np.exp(G_r)
    c_r = g_r - G_r - 1
    return c_r, g_r


def percus_yevick(U, G_r):
    """the PY closure, calculates the direct correlation function from the pair potential and
       the gamma function; also the total and pair correlation functions

       g_r = exp(-U) * (1 + G_r)
       h_r = g_r - 1
       c_r = exp(-U) * (1 + G_r) - G_r - 1

       the discontinuities of the interaction potentials are taken care of here as well
       with help of the U_discontinuity list
    """
    g_r = np.exp(-U.ij) * np.exp(U.erf_real) * (1 + G_r)
    c_r = g_r - G_r - 1
    return c_r, g_r


supported_closures = {'hnc': hypernetted_chain,
                      'py': percus_yevick}
