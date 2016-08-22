from copy import deepcopy
import time

import numpy as np


import pyoz as oz
from pyoz.closure import supported_closures
from pyoz.exceptions import PyozError
from pyoz import dft as ft
from pyoz.misc import rms_normed, solver
from pyoz.potentials import TotalPotential
import pyoz.unit as u
from pyoz.unit import AVOGADRO_CONSTANT_NA as Na
from pyoz.unit import BOLTZMANN_CONSTANT_kB as kB


def prep_input(inputs):
    settings = deepcopy(oz.defaults)
    settings.update(deepcopy(inputs))

    settings['n_points'] -= 1
    return settings


class Component(object):
    def __init__(self, name, concentration=1 * u.moles / u.liters):
        self.name = name
        self._concentration = None
        self.concentration = concentration
        self.potentials = set()

    @property
    def concentration(self):
        return self._concentration

    @concentration.setter
    def concentration(self, c):
        # TODO: units
        if c._value < 0:
            raise PyozError('Concentrations must be >= 0')
        self._concentration = (c * Na).in_units_of(u.angstroms**-3)

    @property
    def n_potentials(self):
        return len(self.potentials)

    @property
    def parameters(self):
        return {pot: pot.parameters.loc[self] for pot in self.potentials}

    def add_potential(self, potential, **parameters):
        self.potentials.add(potential)
        potential.add_component(self, **parameters)

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

        # Physical Constants
        # ==================
        self.T = kwargs.get('T') or 300 * u.kelvin
        self.eps_r = kwargs.get('eps_r') or 78.3 * u.dimensionless
        self.eps_0 = (kwargs.get('eps_0') or
                      8.854187817e-12 * u.farad / u.meter)
        self.kT = self.T * kB

        # Coulomb interaction factor - Bjerrum length
        # V(coul) in kT is then calculated as V = b_l * z1 * z2 / r
        # with z in elementary charge units and r in A
        coul_factor = 4 * np.pi * self.eps_0 * self.eps_r * self.kT
        self.bjerrum_length = ((1 * u.elementary_charge)**2 /
                               coul_factor).value_in_unit(u.angstroms)

        # Algorithm control
        # =================
        n_points_exp = kwargs.get('n_points_exp') or 12
        self.n_points = 2**n_points_exp - 1

        self.dr = kwargs.get('dr') or 0.05 * u.angstrom

        max_r = self.dr.value_in_unit(u.angstrom) * self.n_points
        self.dk = np.pi / max_r

        self.iteration_scheme = kwargs.get('iteration_scheme') or 'picard'
        self.mix_param = kwargs.get('mix_param') or 0.5
        self.tol = kwargs.get('tol') or 1e-9
        self.max_iter = kwargs.get('max_iter') or 1000
        self.max_dsqn = kwargs.get('max_dsqn') or 100.0

        # System info
        # ===========
        self.closure = kwargs.get('closure') or 'hnc'

        # TODO: units
        dr = self.dr = self.dr.value_in_unit(u.angstroms)
        dk = self.dk
        self.r = np.linspace(dr, self.n_points * dr - dr, self.n_points)
        self.k = np.linspace(dk, self.n_points * dk - dk, self.n_points)

        self.components = list()

        # Results get stored after `System.solve` successfully completes.
        self.g_r = None
        self.h_r = None
        self.c_r = None
        self.G_r = None
        self.S_k = None

    @property
    def n_components(self):
        return len(self.components)

    @property
    def potentials(self):
        return {p for c in self.components for p in c.potentials}

    def add_component(self, component):
        self.components.append(component)
        for potential in component.potentials:
            self.potentials.add(potential)

    def _apply_potentials(self):
        for potential in self.potentials:
            potential.apply()
        self.U_r = TotalPotential(r=self.r, n_components=len(self.components),
                                  potentials=self.potentials)

    def solve(self, closure='hnc', status_updates=True):
        self.g_r = self.h_r = self.c_r = self.G_r = self.S_k = None

        self._apply_potentials()
        n_components = self.n_components
        n_points = self.n_points

        # TODO: simplify and units
        concs = [comp.concentration for comp in self.components]
        rho = np.zeros(shape=(n_components, n_components))
        for i, j in np.ndindex(rho.shape):
            rho[i, j] = np.sqrt(concs[i]._value * concs[j]._value)
        self.rho = rho

        matrix_shape = (n_components, n_components, n_points)
        dft = ft.dft(n_points, self.dr, self.dk, self.r, self.k)

        logger = oz.logger
        logger.info('Initialized: {}'.format(self))
        logger.info('Components:')
        for comp in self.components:
            logger.info('   {}'.format(comp))
        logger.info('')

        U_r = self.U_r
        G_r = -U_r.erf_real

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
            c_r, g_r = closure(U_r, G_r)

            # Take us to fourier space.
            for i, j in np.ndindex(n_components, n_components):
                Cs_k[i, j], C_k[i, j] = dft.dfbt(c_r[i, j],
                                                 norm=self.rho[i, j],
                                                 corr=-U_r.erf_fourier[i, j])

            # Solve dat equation.
            A = E - dft.ft_convolution_factor * C_k
            B = C_k
            H_k = solver(A, B)
            S_k = E + H_k
            G_k = S_k - E - Cs_k

            # Snap back to reality.
            for i, j in np.ndindex(n_components, n_components):
                G_r[i, j] = dft.idfbt(G_k[i, j],
                                      norm=self.rho[i, j],
                                      corr=-U_r.erf_real[i, j])

            # Test for convergence.
            rms_norm = rms_normed(G_r, G_r_previous)

            if rms_norm < self.tol:
                converged = True
                break

            # Iterate.
            if self.iteration_scheme == 'picard':
                mix = self.mix_param
                G_r = (1 - mix) * G_r_previous + mix * G_r
            else:
                raise PyozError('Iteration scheme "{}" not yet '
                                'implemented.'.format(self.iteration_scheme))
            if status_updates:
                time_taken = time.time() - loop_start
                logger.info('   {:<8d}{:<8.2f}{:<8.2e}'.format(
                    n_iter, time_taken, rms_norm)
                )
        end = time.time()
        if converged:
            logger.info('Converged in {:.2f}s after {} iterations'.format(
                end-start, n_iter)
            )
            c_r, g_r = closure(U_r, G_r)
            self.g_r = g_r
            self.h_r = g_r - 1
            self.c_r = c_r
            self.G_r = G_r
            self.S_k = S_k
            return

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
