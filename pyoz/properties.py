from collections import OrderedDict

import numpy as np
from scipy.integrate import cumtrapz as integrate

from pyoz.closure import hypernetted_chain, reference_hypernetted_chain
from pyoz.exceptions import PyozError


__all__ = ['kirkwood_buff_integrals',
           'structure_factors',
           'excess_chemical_potential',
           'pressure_virial',
           'second_virial_coefficient',
           'two_particle_excess_entropy']


def kirkwood_buff_integrals(system):
    """Compute the Kirkwood-Buff integrals.

    G_ij = 4 pi \int_0^\inf [g(r)-1]r^2 dr
    """
    r, g_r = system.r, system.g_r
    return 4.0 * np.pi * integrate(y=(g_r - 1.0) * r**2,
                                   x=r,
                                   even='last')


def structure_factors(system, formalism='Faber-Ziman',
                      combination='number-number'):
    """Compute the partial structure factors.

    Parameters
    ----------
    system : pyoz.System
        The solved system for which to compute structure factors.
    formalism : str, optional, default='Faber-Ziman'
        The formalism used to compute the structure factor. Supported values
        are 'Faber-Ziman', 'Ashcroft-Langreth' and 'Bhatia-Thornton'
    combination : str, optional, default='number-number'
        When using the Bhatia-Thornton formalism, specifies whether to return
        the number-number, number-concentration or concentration-concentration
        partial structure factors.

    Returns
    -------
    S_k : np.ndarray, shape=(n_components, n_components, n_pts)
        The partial structure factors for each species.

    References
    ----------
    .. [1] http://isaacs.sourceforge.net/phys/scatt.html

    """
    try:
        S_k_function = _sk_formalisms[formalism.lower()]
    except KeyError:
        keys = '\t'.join(['"{}"\n'.format(x) for x in _sk_formalisms.keys()])
        raise PyozError('Unsupported structure factor formalism. Valid options '
                        'are:\n \t{}'.format(keys))
    return S_k_function(system, combination)


def _faber_ziman(system, combination):
    h_k = system.h_k
    return 1 + h_k


def _ashcroft_langreth(system, combination):
    h_k = system.h_k
    E = np.zeros_like(h_k)
    for n in range(h_k.shape[2]):
        E[:, :, n] = np.eye(h_k.shape[0])
    return E + h_k


def _bhatia_thornton(system, combination):
    if system.h_k.shape[0] != 2:
        raise NotImplementedError('Only implemented for two component systems')
    try:
        Sxx_function = _bhatia_thornton_combinations[combination.lower()]
    except KeyError:
        keys = '\t'.join(['"{}"\n'.format(x)
                          for x in _bhatia_thornton_combinations.keys()])
        raise PyozError('Unsupported combination for Bhatia-Thornton formalism.'
                        ' Valid options are:\n \t{}'.format(keys))
    return Sxx_function(system)


def _Snn(system):
    rhos, h_k = np.diag(system.rho_ij), system.h_k
    rho = np.sum(rhos)
    xs = rhos / rho
    return 1 + rho * (    xs[0] * xs[0] * h_k[0, 0] +
                      2 * xs[0] * xs[1] * h_k[0, 1] +
                          xs[1] * xs[1] * h_k[1, 1])


def _Snc(system):
    rhos, h_k = np.diag(system.rho_ij), system.h_k
    rho = np.sum(rhos)
    xs = rhos / rho
    x_ij = xs[0] * xs[1]
    return rho * x_ij * (xs[0] * (h_k[0, 0] - h_k[0, 1]) -
                         xs[1] * (h_k[1, 1] - h_k[0, 1]))


def _Scc(system):
    rhos, h_k = np.diag(system.rho_ij), system.h_k
    rho = np.sum(rhos)
    xs = rhos / rho
    x_ij = xs[0] * xs[1]
    return x_ij * (1 + rho * x_ij * (    h_k[0, 0] +
                                         h_k[1, 1] -
                                     2 * h_k[0, 1]))

_sk_formalisms = OrderedDict([('faber-ziman', _faber_ziman),
                              ('fz', _faber_ziman),
                              ('ashcroft-langreth', _ashcroft_langreth),
                              ('al', _ashcroft_langreth),
                              ('bhatia-thornton', _bhatia_thornton),
                              ('bt', _bhatia_thornton),
])

_bhatia_thornton_combinations = OrderedDict([('number-number', _Snn),
                                             ('nn', _Snn),
                                             ('number-concentration', _Snc),
                                             ('nc', _Snc),
                                             ('concentration-number', _Snc),
                                             ('cn', _Snc),
                                             ('concentration-concentration', _Scc),
                                             ('cc', _Scc),
])


def pressure_virial(system):
    """Compute the pressure via the virial route

    P = \rho * \beta - 2/3 * pi * int_0^inf [(r*dU/dr) * g(r) * r^2]dr

    """
    r, g_r, U_r, rho_ij, kT = system.r, system.g_r, system.U_r, system.rho_ij, system.kT
    dr = r[1] - r[0]
    dUdr = (np.diff(U_r) / dr)

    integral = integrate(y=r[1:]**3 * g_r[:, :, 1:] * dUdr, x=r[1:])

    rhos = np.diag(rho_ij)
    rho = np.sum(rhos)

    pressure = rho * kT
    for i, j in np.ndindex(rho_ij.shape):
       pressure -= 2/3 * np.pi * rhos[i] * rhos[j] * integral[i, j]
    return pressure


def excess_chemical_potential(system):
    """Compute the excess chemical potentials.

    \beta mu_i^{ex} = \sum_i 4 \pi \rho_i  *
                            \int [ h(r) * e(r) / 2 - c^s(r) ] r^2 dr

    Only valid for the HNC closure.

    """
    if system.closure_used not in (hypernetted_chain,
                                   reference_hypernetted_chain):
        raise PyozError('Excess chemical potential calculation is only valid'
                        'for hyper-netted chain closures.')
    r, h_r, e_r, c_r, kT = system.r, system.h_r, system.e_r, system.c_r, system.kT
    rho = system.rho_ij
    n_components = system.n_components
    mu_ex = np.zeros(shape=n_components)
    for i in range(n_components):
        for j in range(n_components):
            integrand = ((h_r[i, j] * e_r[i, j]) / 2 - c_r[i, j]) * r**2
            mu_ex[i] += 4.0 * np.pi * rho[j] * integrate(y=integrand,
                                                         x=r,
                                                         even='last')
    return mu_ex * kT


def second_virial_coefficient(system):
    r, U_r, kT, rho = system.r, system.U_r, system.kT, system.rho_ij
    if U_r.shape[0] == 1:
        return -2 * np.pi * integrate(y=(np.exp(-U_r[0, 0] / kT) - 1) * r**2, x=r)
    elif U_r.shape[0] == 2:
        x = rho.diagonal() / rho.diagonal().sum()
        B2 = 0
        for i, j in np.ndindex(U_r.shape[:2]):
            U_ij = U_r[i, j]
            B2_ij = -2 * np.pi * integrate(y=(np.exp(-U_ij / kT) - 1) * r**2, x=r)
            B2 += x[i] * x[j] * B2_ij
        return B2
    else:
        raise NotImplementedError('Virial calculation not yet implemented for '
                                  'systems with more than two components.')


def two_particle_excess_entropy(system):
    """Compute 2-particle excess entropy.

    Eqn. 9 in A Baranyi and DJ Evans, Phys. Rev. A., 1989
    """
    r, g_r, rho = system.r, system.g_r, system.rho_ij[0]
    if g_r.shape[0] > 1:
        raise NotImplementedError('Entropy calculation not yet '
                                  'implemented for multi-component systems.')
    g_r = g_r[0, 0]
    integrand = np.where(g_r > 0,
                         -0.5 * rho * (g_r * np.log(g_r) - g_r + 1.0),
                         -0.5 * rho)
    return rho * integrate(integrand, r)


def isothermal_compressibility(system):
    if system.g_r.shape[0] > 1:
        raise NotImplementedError('Compressibility calculation not yet '
                                  'implemented for multi-component systems.')
    return system.S_k[0] / system.rho_ij[0] / system.kT


def activity_coefficient(system):
    """Compute the activity coefficients.

    \gamma_i = exp(\beta \mu^ex)

    """
    # TODO: add mean activity calculation for charged system
    mu_ex = excess_chemical_potential(system)
    return np.exp(mu_ex / system.kT)

