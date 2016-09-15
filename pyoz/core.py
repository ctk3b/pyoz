import time

import numpy as np


import pyoz as oz
from pyoz.closure import supported_closures
from pyoz.exceptions import PyozError
from pyoz import dft as ft
from pyoz.misc import rms_normed, solver, picard_iteration
import pyoz.unit as u
from pyoz.unit import BOLTZMANN_CONSTANT_kB as kB


class System(object):
    def __init__(self, name='System', **kwargs):
        self.name = name

        # Physical Constants
        # ==================
        self.T = kwargs.get('T') or 1
        self.eps_r = kwargs.get('eps_r') or 78.3 * u.dimensionless
        self.eps_0 = (kwargs.get('eps_0') or
                      8.854187817e-12 * u.farad / u.meter)
        self.kT = self.T * u.kelvin * kB

        # Coulomb interaction factor - Bjerrum length
        # V(coul) in kT is then calculated as V = b_l * z1 * z2 / r
        # with z in elementary charge units and r in A
        coul_factor = 4 * np.pi * self.eps_0 * self.eps_r * self.kT
        self.bjerrum_length = ((1 * u.elementary_charge)**2 /
                               coul_factor).value_in_unit(u.angstroms)

        # Algorithm control
        # =================
        n_points_exp = kwargs.get('n_points_exp') or 12
        self.n_points = kwargs.get('n_points') or 2**n_points_exp
        self.n_points -= 1

        self.dr = kwargs.get('dr') or 0.01

        max_r = self.dr * self.n_points
        self.dk = np.pi / max_r

        # System info
        # ===========
        self.closure = kwargs.get('closure') or 'hnc'

        # TODO: units
        dr = self.dr
        dk = self.dk
        self.r = np.linspace(dr, self.n_points * dr - dr, self.n_points)
        self.k = np.linspace(dk, self.n_points * dk - dk, self.n_points)
        self.U_r = np.zeros(shape=(0, 0, self.n_points))
        self.U_r_erf_fourier = np.zeros(shape=(0, 0, self.n_points))
        self.U_r_erf_real = np.zeros(shape=(0, 0, self.n_points))
        self.rho = None

        # Results get stored after `System.solve` successfully completes.
        self.g_r = self.h_r = self.c_r = self.G_r = self.S_k = None

    @property
    def n_components(self):
        return self.U_r.shape[0]

    def set_interaction(self, comp1_idx, comp2_idx, potential,
                        long_range_real=None, long_range_fourier=None,
                        symmetric=True):
        if len(potential) != self.n_points:
            raise PyozError('Attempted to add values at {} points to potential'
                            'with {} points.'.format(potential.shape, self.n_points))
        if comp1_idx >= self.n_components or comp2_idx >= self.n_components:
            n_bigger = max(comp1_idx, comp2_idx) - self.U_r.shape[0] + 1
            self.U_r.resize((self.U_r.shape[0] + n_bigger,
                             self.U_r.shape[1] + n_bigger,
                             self.n_points))
            self.U_r_erf_real.resize((self.U_r_erf_real.shape[0] + n_bigger,
                                      self.U_r_erf_real.shape[1] + n_bigger,
                                      self.n_points))
            self.U_r_erf_fourier.resize((self.U_r_erf_fourier.shape[0] + n_bigger,
                                         self.U_r_erf_fourier.shape[1] + n_bigger,
                                         self.n_points))
        self.U_r[comp1_idx, comp2_idx] = potential
        if symmetric and comp1_idx != comp2_idx:
            self.U_r[comp2_idx, comp1_idx] = potential

        if ((long_range_real is None and long_range_fourier is not None) or
                (long_range_real is not None and long_range_fourier is None)):
            raise PyozError('Provided long-range potential for either real'
                            'or fourier space but not both')
        if long_range_real is None:
            long_range_real = np.zeros_like(potential)
            long_range_fourier = np.zeros_like(potential)

        self.U_r_erf_real[comp1_idx, comp2_idx] = long_range_real
        self.U_r_erf_fourier[comp1_idx, comp2_idx] = long_range_fourier

        if symmetric and comp1_idx != comp2_idx:
            self.U_r[comp2_idx, comp1_idx] = potential
            self.U_r_erf_real[comp2_idx, comp1_idx] = long_range_real
            self.U_r_erf_fourier[comp2_idx, comp1_idx] = long_range_fourier

    def remove_interaction(self, comp1_idx, comp2_idx):
        # Needs to reduce size of U_r if comp1_idx == comp2_idx
        raise NotImplementedError

    def solve(self, rhos, closure_name='hnc', initial_G_r=None,
              status_updates=True, iteration_scheme='picard', mix_param=0.8,
              tol=1e-9, max_iter=1000, **kwargs):
        if self.U_r.shape[0] == 0:
            raise PyozError('No interactions to solve. Use `add_interaction`'
                            'before calling `solve`.')
        if not hasattr(rhos, '__iter__'):
            rhos = [rhos]
        if self.U_r.shape[0] != len(rhos):
            raise PyozError("Number of ρ's provided does not match dimensions"
                            " of potential")

        if closure_name.upper() == 'RHNC':
            if kwargs.get('g_r_ref') is None:
                raise PyozError('Missing `g_r_ref` parameter for RHNC closure.')
            if kwargs.get('G_r_ref') is None:
                raise PyozError('Missing `G_r_ref` parameter for RHNC closure.')
            if kwargs.get('U_r_ref') is None:
                raise PyozError('Missing `U_r_ref` parameter for RHNC closure.')

        try:
            closure = supported_closures[closure_name.upper()]
        except KeyError:
            raise PyozError('Unsupported closure: ', closure_name)

        self.g_r = self.h_r = self.c_r = self.G_r = self.S_k = None

        U_r, U_r_erf_real, U_r_erf_fourier = self.U_r, self.U_r_erf_real, self.U_r_erf_fourier
        n_components = self.n_components
        n_points = self.n_points

        rho = np.zeros(shape=(n_components, n_components))
        for i, j in np.ndindex(rho.shape):
            rho_ij = np.sqrt(rhos[i] * rhos[j])
            U_r_erf_fourier[i, j] = U_r_erf_fourier[i, j] * rho_ij
            rho[i, j] = rho_ij
        self.rho = rho

        matrix_shape = (n_components, n_components, n_points)
        dft = ft.dft(n_points, self.dr, self.dk, self.r, self.k)

        logger = oz.logger
        if status_updates:
            logger.info('Initialized: {}'.format(self))

        if initial_G_r is None:
            Gs_r = -U_r_erf_real
        else:
            Gs_r = initial_G_r

        C_k = np.zeros(matrix_shape)
        Cs_k = np.zeros(matrix_shape)

        E = np.zeros(matrix_shape)
        for n in range(n_points):
            E[:, :, n] = np.eye(n_components)

        converged = False
        total_iter = 0
        n_iter = 0
        start = time.time()

        if status_updates:
            logger.info('Starting iteration...')
            logger.info('   {:8s}{:10s}{:10s}'.format(
                'step', 'time (s)', 'error'))
        while not converged and n_iter < max_iter:
            loop_start = time.time()
            n_iter += 1
            total_iter += 1
            Gs_r_previous = np.copy(Gs_r)

            # Apply the closure relation.
            cs_r = closure(U_r, Gs_r, U_r_erf_real, **kwargs)

            # Take us to fourier space.
            for i, j in np.ndindex(n_components, n_components):
                Cs_k[i, j], C_k[i, j] = dft.dfbt(cs_r[i, j],
                                                 norm=self.rho[i, j],
                                                 corr=-U_r_erf_fourier[i, j])

            # Solve dat equation.
            A = E - dft.ft_convolution_factor * C_k
            B = C_k
            H_k = solver(A, B)
            S_k = 1 + H_k
            Gs_k = H_k - Cs_k

            # Snap back to reality.
            for i, j in np.ndindex(n_components, n_components):
                Gs_r[i, j] = dft.idfbt(Gs_k[i, j],
                                      norm=self.rho[i, j],
                                      corr=U_r_erf_real[i, j])

            # Test for convergence.
            rms_norm = rms_normed(Gs_r, Gs_r_previous)
            if rms_norm < tol:
                converged = True
                break
            elif np.isnan(rms_norm):
                raise PyozError('Diverged at iteration # {}'.format(n_iter))

            # Iterate.
            if iteration_scheme == 'picard':
                Gs_r = picard_iteration(Gs_r, Gs_r_previous, mix_param)
            else:
                raise PyozError('Iteration scheme "{}" not yet '
                                'implemented.'.format(iteration_scheme))
            if status_updates:
                time_taken = time.time() - loop_start
                logger.info('   {:<8d}{:<8.2f}{:<8.2e}'.format(
                    n_iter, time_taken, rms_norm)
                )
        end = time.time()
        if converged:
            # Set before error so you can still extract info if unphysical.
            cs_r = closure(U_r, Gs_r, U_r_erf_real, **kwargs)
            # import ipdb; ipdb.set_trace()
            G_r = Gs_r + U_r_erf_real
            self.c_r = c_r = cs_r - U_r_erf_real
            self.g_r = g_r = c_r + G_r + 1
            self.h_r = g_r - 1
            self.G_r = G_r
            self.S_k = S_k

            if (S_k < 0).any():
                raise PyozError('Converged to unphysical result.')

            logger.info('Converged in {:.2f}s after {} iterations'.format(
                end-start, n_iter)
            )
            return g_r, c_r, G_r, S_k

        raise PyozError('Exceeded max # of iterations: {}'.format(n_iter))

    def __repr__(self):
        descr = list('<{}'.format(self.name))
        if self.rho is not None:
            descr.append('; {} component'.format(self.rho.shape[0]))
            if self.rho.shape[0] > 1:
                descr.append('s')
            descr.append('; ρ:')
            for rho in self.rho.diagonal():
                descr.append(' {:.6g}'.format(rho))
        descr.append('>')
        return ''.join(descr)
