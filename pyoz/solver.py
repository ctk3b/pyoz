from copy import deepcopy
import itertools as it
import time

import numpy as np
import simtk.unit as u
from simtk.unit import AVOGADRO_CONSTANT_NA as Na
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB

import pyoz
from pyoz.closure import supported_closures
from pyoz import dft as ft
from pyoz.exceptions import PyozError
from pyoz.misc import rms_normed
from pyoz.potential import Potential


def prep_input(input_dict):
    settings = deepcopy(pyoz.defaults)
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
    dens = np.zeros(shape=(n_components, n_components))
    for i, j in it.product(range(n_components), range(n_components)):
        dens[i, j] = np.sqrt(concs[i]._value * concs[j]._value)
    settings['dens'] = dens

    return settings


def solve_ornstein_zernike(inputs, status_updates=True):
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
    G_r = -U.erf_real

    # c(r) with density factor applied: C(r)
    # c(k) with density factor applied: C(k)
    C_k = np.zeros(shape=(n_components, n_components, n_points))
    Cs_k = np.zeros(shape=(n_components, n_components, n_points))

    E = np.zeros(shape=(n_components, n_components, n_points))
    for n in range(n_points):
        E[:, :, n] = np.eye(n_components)

    closure = supported_closures[settings['closure']]

    converged = False
    total_iter = 0
    n_iter = 0
    start = time.time()

    logger.info('Starting iteration...')
    if status_updates:
        logger.info('   {:8s}{:10s}{:10s}'.format('step', 'time (s)', 'error'))
    while not converged and n_iter < settings['max_iter']:
        loop_start = time.time()
        n_iter += 1
        total_iter += 1
        G_r_previous = np.copy(G_r)

        # Apply the closure relation.
        c_r, g_r = closure(U, G_r)

        # Take us to fourier space.
        for i, j in it.product(range(n_components), range(n_components)):
            Cs_k[i, j], C_k[i, j] = dft.dfbt(c_r[i, j],
                                             norm=settings['dens'][i, j],
                                             corr=-U.erf_fourier[i, j])

        # Solve dat equation.
        A = E - dft.ft_convolution_factor * C_k
        B = C_k
        H_k = np.empty_like(A)
        for dr in range(n_points):
            H_k[:, :, dr] = np.linalg.solve(A[:, :, dr], B[:, :, dr])

        S = E + H_k
        G_k = S - E - Cs_k

        # Snap back to reality.
        for i, j in it.product(range(n_components), range(n_components)):
            G_r[i, j] = dft.idfbt(G_k[i, j],
                                  norm=settings['dens'][i, j],
                                  corr=-U.erf_real[i, j])

        # Test for convergence.
        norm_dsqn = rms_normed(G_r, G_r_previous)

        if norm_dsqn < settings['tol']:
            converged = True
            break

        # Iterate.
        iter_scheme = settings['iteration_scheme']
        if iter_scheme == 'picard':
            mix = settings['mix_param']
            G_r = (1 - mix) * G_r_previous + mix * G_r
        else:
            raise PyozError('Iteration scheme "{}" not yet '
                            'implemented.'.format(iter_scheme))
        if status_updates:
            logger.info('   {:<8d}{:<8.2f}{:<8.2e}'.format(n_iter,
                                                        time.time() - loop_start,
                                                        norm_dsqn))
    end = time.time()
    if converged:
        logger.info('Converged in {:.2f}s after {} iterations'.format(end-start, n_iter))
        c_r, g_r = closure(U, G_r)
        return r, g_r
    else:
        raise PyozError('Exceeded max # of iterations: {}'.format(n_iter))
