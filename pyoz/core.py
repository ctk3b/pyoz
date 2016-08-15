from copy import deepcopy
import itertools as it
import time

import numpy as np
import simtk.unit as u
from simtk.unit import AVOGADRO_CONSTANT_NA as Na
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB

import pyoz as oz
from pyoz.closure import supported_closures
from pyoz.exceptions import PyozError
from pyoz import dft as ft
from pyoz import logger
from pyoz.misc import rms_normed
from pyoz.potentials import TotalPotential


def prep_input(inputs):
    settings = deepcopy(oz.defaults)
    settings.update(deepcopy(inputs))

    settings['n_points'] -= 1
    return settings


class Component(object):
    def __init__(self, name, concentration):
        self.name = name
        self.concentration = concentration
        self.potentials = list()

    @property
    def concentration(self):
        return self._concentration

    @concentration.setter
    def concentration(self, c):
        # TODO: units
        self._concentration = (c * Na).in_units_of(u.angstroms**-3)

    def add_potential(self, potential, parameters):
        potential.add_parms(self, **parameters)
        self.potentials.append(potential)

    def __repr__(self):
        descr = list('<{}, '.format(self.name))
        descr.append('rho: {}, '.format(
            self.concentration.format('%8.8f')))
        descr.append('potentials:')
        if self.potentials:
            n_potentials = len(self.potentials)
            for n, pot in enumerate(self.potentials):
                descr.append(' {}'.format(pot))
                if n < n_potentials - 1:
                    descr.append(' +')
        else:
            descr.append('None')
        descr.append('>')
        return ''.join(descr)


class System(object):
    def __init__(self, name='System', **kwargs):
        self.name = name
        settings = prep_input(kwargs)
        for attribute, value in settings.items():
            setattr(self, attribute, value)

        # TODO: units
        dr = self.dr = self.dr.value_in_unit(u.angstroms)
        dk = self.dk
        n_points = settings['n_points']
        self.r = np.linspace(dr, n_points * dr - dr, n_points)
        self.k = np.linspace(dk, n_points * dk - dk, n_points)

        self.components = list()
        self.potentials = set()

    @property
    def n_components(self):
        return len(self.components)

    def add_component(self, component):
        self.components.append(component)
        for potential in component.potentials:
            self.potentials.add(potential)

    def _apply_potentials(self):
        for potential in self.potentials:
            potential.apply(self.r, self.T)
        self.U = TotalPotential(r=self.r, n_components=len(self.components),
                                potentials=self.potentials)

    def solve(self, closure='hnc', status_updates=True):
        self._apply_potentials()
        n_components = self.n_components
        n_points = self.n_points

        # TODO
        concs = [comp.concentration for comp in self.components]
        dens = np.zeros(shape=(n_components, n_components))
        for (i, j), _ in np.ndenumerate(dens):
            dens[i, j] = np.sqrt(concs[i]._value * concs[j]._value)
        self.dens = dens

        matrix_shape = (n_components, n_components, n_points)
        dft = ft.dft(n_points, self.dr, self.dk, self.r, self.k)

        logger.info('Initialized: {}'.format(self))
        logger.info('Components:')
        for comp in self.components:
            logger.info('   {}'.format(comp))
        logger.info('')

        U = self.U
        G_r = -U.erf_real

        # c(r) with density factor applied: C(r)
        # c(k) with density factor applied: C(k)
        C_k = np.zeros(matrix_shape)
        Cs_k = np.zeros(matrix_shape)

        E = np.zeros(matrix_shape)
        for n in range(n_points):
            E[:, :, n] = np.eye(n_components)

        closure = supported_closures[closure]

        converged = False
        total_iter = 0
        n_iter = 0
        start = time.time()

        logger.info('Starting iteration...')
        if status_updates:
            logger.info('   {:8s}{:10s}{:10s}'.format(
                'step', 'time (s)', 'error'))
        while not converged and n_iter < self.max_iter:
            loop_start = time.time()
            n_iter += 1
            total_iter += 1
            G_r_previous = np.copy(G_r)

            # Apply the closure relation.
            c_r, g_r = closure(U, G_r)

            # Take us to fourier space.
            for i, j in it.product(range(n_components), range(n_components)):
                Cs_k[i, j], C_k[i, j] = dft.dfbt(c_r[i, j],
                                                 norm=self.dens[i, j],
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
                                      norm=self.dens[i, j],
                                      corr=-U.erf_real[i, j])

            # Test for convergence.
            rms_norm = rms_normed(G_r, G_r_previous)

            if rms_norm < self.tol:
                converged = True
                break

            # Iterate.
            iter_scheme = self.iteration_scheme
            if iter_scheme == 'picard':
                mix = self.mix_param
                G_r = (1 - mix) * G_r_previous + mix * G_r
            else:
                raise PyozError('Iteration scheme "{}" not yet '
                                'implemented.'.format(iter_scheme))
            if status_updates:
                logger.info('   {:<8d}{:<8.2f}{:<8.2e}'.format(n_iter,
                                                            time.time() - loop_start,
                                                            rms_norm))
        end = time.time()
        if converged:
            logger.info('Converged in {:.2f}s after {} iterations'.format(end-start, n_iter))
            c_r, g_r = closure(U, G_r)
            return self.r, g_r

        raise PyozError('Exceeded max # of iterations: {}'.format(n_iter))

    def __repr__(self):
        descr = list('<{}, '.format(self.name))
        descr.append('components: ')
        for comp in self.components:
            descr.append('{}, '.format(comp.name))
        descr.append('potentials:')
        n_potentials = len(self.potentials)
        for n, pot in enumerate(self.potentials):
            descr.append(' {}'.format(pot))
            if n < n_potentials - 1:
                descr.append(' +')
        descr.append('>')
        return ''.join(descr)
