#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining closure relations
"""

import numpy as np


def hypernetted_chain(U, G_r_ij):
    """the HNC closure, calculates the direct correlation function from the pair potential and
       the gamma function; also the total and pair correlation functions

       according to HNC
       g_ij = modMayerFunc*exp(G_r_ij)
       h_ij = g_ij - 1
       c_ij = modMayerFunc*exp(G_r_ij) - G_r_ij - 1

       the discontinuities of the interaction potentials are taken care of here as well
       with help of the U_discontinuity list
    """

    g_r_ij = np.exp(-U.ij) * np.exp(U.erf_ij_real) * np.exp(G_r_ij)
    c_r_ij = g_r_ij - G_r_ij - 1

    return c_r_ij, g_r_ij


def percus_yevick(U, G_r_ij):
    """the PY closure, calculates the direct correlation function from the pair potential and
       the gamma function; also the total and pair correlation functions

       according to PY
       g_ij = exp(-U_ij) * (1 + G_r_ij)
       h_ij = g_ij - 1
       c_ij = exp(-U_ij) * (1 + G_r_ij) - G_r_ij - 1

       the discontinuities of the interaction potentials are taken care of here as well
       with help of the U_discontinuity list
    """
    g_r_ij = np.exp(-U.ij) * np.exp(U.erf_ij_real) * (1 + G_r_ij)
    c_r_ij = g_r_ij - G_r_ij - 1

    return c_r_ij, g_r_ij


supported_closures = {'hnc': hypernetted_chain,
                      'py': percus_yevick}
