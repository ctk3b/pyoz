from copy import deepcopy
import itertools as it
import os
from time import time

import numpy as np
import simtk.unit as u
from simtk.unit import AVOGADRO_CONSTANT_NA as Na
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB
import yaml

import pyoz
from pyoz.potential import Potential
import pyoz.thermodynamic_properties as properties

from pyoz import dft as ft
from pyoz.closure import supported_closures
from pyoz.misc import squared_normed_distance, dotproduct


def prep_input(input_dict):
    settings = deepcopy(pyoz.settings)
    settings.update(deepcopy(input_dict))

    settings['n_points'] -= 1

    if settings['potentials']['lennard-jones']:
        lj_parms = settings['potentials']['lennard-jones']
        # TODO: General unit handling
        lj_parms['sigmas'] = np.array([x.in_units_of(u.angstroms)
                                      for x in lj_parms['sigmas']])
        lj_parms['epsilons'] = np.array([(x / Na).in_units_of(u.joule) / kB / settings['T']
                                        for x in lj_parms['epsilons']])

    # Convert to particles per A^3
    settings['concentrations'] = np.array([(x * Na).in_units_of(u.angstrom**-3)
                                           for x in settings['concentrations']])
    concs = settings['concentrations']
    n_components = settings['n_components']
    dens_ij = np.zeros(shape=(n_components, n_components))
    for i, j in it.product(range(n_components), range(n_components)):
        dens_ij[i, j] = np.sqrt(concs[i]._value * concs[j]._value)
    settings['dens_ij'] = dens_ij

    return settings


def solve_ornstein_zernike(inputs):
    logger = pyoz.logger
    settings = prep_input(inputs)

    dr = settings['dr'].value_in_unit(u.angstroms)
    dk = settings['dk']
    n_points = settings['n_points']
    r = np.linspace(dr, n_points * dr - dr, n_points)
    k = np.linspace(dk, n_points * dk - dk, n_points)
    dft = ft.dft(n_points, dr, dk, r, k)

    n_components = settings['n_components']
    U = Potential(r,
                  n_components=n_components,
                  potentials=settings['potentials'])

    # Zero Gamma function
    # TODO: custom initial guesses for Gamma
    G_r_ij = -U.erf_ij_real

    # c(r) with density factor applied: C(r)
    # c(k) with density factor applied: C(k)
    C_k_ij = np.zeros(shape=(n_components, n_components, n_points))
    Cs_k_ij = np.zeros(shape=(n_components, n_components, n_points))

    E_ij = np.zeros(shape=(n_components, n_components, n_points))
    for n in range(n_points):
        E_ij[:, :, n] = np.eye(n_components)

    closure = supported_closures[settings['closure']]

    converged = False
    total_iter = 0
    n_iter = 0
    while not converged and n_iter < settings['max_iter']:
        n_iter += 1
        total_iter += 1
        G_r_ij_previous = np.copy(G_r_ij)

        # Apply the closure relation.
        cs_r_ij, g_r_ij = closure(U, G_r_ij)

        # Take us to fourier space.
        for i, j in it.product(range(n_components), range(n_components)):
            Cs_k_ij[i, j], C_k_ij[i, j] = dft.dfbt(cs_r_ij[i, j],
                                                   norm=settings['dens_ij'][i, j],
                                                   corr=-U.erf_ij_fourier[i, j])

        A = E_ij - dft.ft_convolution_factor * C_k_ij
        B = C_k_ij
        H_k_ij = np.empty_like(A)
        for dr in range(n_points):
            H_k_ij[:, :, dr] = np.linalg.solve(A[:, :, dr], B[:, :, dr])

        S = E_ij + H_k_ij
        G_k_ij = S - E_ij - Cs_k_ij

        # Snap back to reality.
        for i, j in it.product(range(n_components), range(n_components)):
            G_r_ij[i, j] = dft.idfbt(G_k_ij[i, j],
                                     norm=settings['dens_ij'][i, j],
                                     corr=-U.erf_ij_real[i, j])

        norm_dsqn = squared_normed_distance(G_r_ij, G_r_ij_previous)
        if norm_dsqn < settings['tol']:
            converged = True
            break

        if not settings.get('iteration-scheme'):
            mix = settings['mix_param']
            G_r_ij = (1 - mix) * G_r_ij_previous + mix * G_r_ij
        else:
            raise ValueError('Iteration schemes not yet implemented.')

    if converged:
        logger.info('Converged after {} iterations'.format(n_iter))
        cs_r_ij, g_r_ij = closure(U, G_r_ij)
        return r, g_r_ij
    else:
        logger.info('Exceeded max # of iterations: {}'.format(n_iter))
