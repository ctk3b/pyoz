import numpy as np
from scipy.integrate import simps as integrate

from pyoz.closure import hypernetted_chain, reference_hypernetted_chain
from pyoz.exceptions import PyozError


__all__ = ['kirkwood_buff_integrals',
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


# TODO: generalize for multi-component
def pressure_virial(system):
    """Compute the pressure via the virial route

    P = \rho * \beta - 2/3 * pi * int_0^inf [(r*dU/dr) * g(r) * r^2]dr

    """
    r, g_r, U_r, rho, kT = system.r, system.g_r, system.U_r, system.rho, system.kT
    if g_r.shape[0] != 1:
        raise NotImplementedError('Pressure calculation not yet implemented '
                                  'for multi-component systems.')
    min_r = 50
    U_r = np.squeeze(U_r)
    g_r = np.squeeze(g_r)[min_r:-1]
    rho = np.squeeze(rho)
    dr = r[1] - r[0]
    dUdr = (np.diff(U_r) / dr)[min_r:]
    r = r[min_r:-1]

    integral = integrate(y=r**3 * g_r * dUdr, x=r)
    return rho * kT - 2/3 * np.pi * rho**2 * integral


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
    rho = system.rho
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
    r, U_r, kT, rho = system.r, system.U_r, system.kT, system.rho
    if U_r.shape[0] == 1:
        return -2 * np.pi * integrate(y=(np.exp(-U_r[0, 0] / kT) - 1) * r**2, x=r)
    elif U_r.shape[0] == 2:
        x = rho.diagonal() / rho.diagonal().sum()
        B2 = 0
        for i, j in np.ndindex(U_r.shape):
            U_ij = U_r[i, j]
            B2_ij = -2 * np.pi * integrate(y=(np.exp(-U_ij / kT) - 1) * r**2, x=r)
            B2 += x[i] * x[j] * B2_ij
        return B2
    else:
        raise NotImplementedError('Virial calculation not yet '
                                  'implemented for systems with more than two '
                                  'components.')


def two_particle_excess_entropy(system):
    """Compute 2-particle excess entropy.

    Eqn. 9 in A Baranyi and DJ Evans, Phys. Rev. A., 1989
    """
    r, g_r, rho = system.r, system.g_r, system.rho[0]
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
    return system.S_k[0] / system.rho[0] / system.kT


def activity_coefficient(system):
    """Compute the activity coefficients.

    \gamma_i = exp(\beta \mu^ex)

    """
    # TODO: add mean activity calculation for charged system
    mu_ex = excess_chemical_potential(system)
    return np.exp(mu_ex / system.kT)


def excess_internal_energy(system):
    """ """
    r, g_r, U_r = system.r, system.g_r, system.U_r
    U_ex = 2 * np.pi * rho * integrate(y=g_r * U_r * r**2,
                                       x=r,
                                       even='last')
    return U_ex


def compressibility(ctrl, syst, const, r, c_sr):
    """
      calculates isothermal compressibility and excess isothermal compressibility
    """

    if syst['dens']['totnum'] == 0.0:
        # infinite dilution
        chi_ex = 1.0
        chi_ex_r = 1.0
        chi = np.inf
        chi_r = 0.0
        chi_id = np.inf
        chi_id_r = 0.0
    else:
        # calculate the prefactor 4 pi / rho
        prefactor = 4.0 * np.pi / syst['dens']['totnum']
        # reciprocal of the excess compressibility
        # initialize to 1
        chi_ex_r = 1.0
        chi_id = 1.0e-7 / (const.kT * syst['dens']['totnum'])
        chi_id_r = 1.0e7 * const.kT * syst['dens']['totnum']
        # perform the calculation
        # chi_ex_r = 1 - 1/\rho \sum_i \sum_j \rho_i \rho_j \int c_sr 4 \pi r^2 dr
        # 4 \pi / \rho is in the prefactor
        # integrate numerically using simpson rule
        # we could use x = r and skip dx, but the spacing is regular so it's probably better to do it this way
        # simspon requires odd number of samples, what we have; just to be sure, we give the option for the
        # even number of samples - for the first interval the trapezoidal rule is used and then simpson for the rest
        for i in range(syst['ncomponents']):
            for j in range(syst['ncomponents']):
                contrib = prefactor * syst['dens']['num'][i] * \
                          syst['dens']['num'][j] * integrate.simps(
                    c_sr[i, j] * r ** 2, r, dx=ctrl['deltar'], even='last')
                chi_ex_r -= contrib
                # print i,j,contrib

        chi_ex = 1.0 / chi_ex_r

        # 1/chi_ex = chi_id/chi; chi_id = 1/\rho kT
        # chi = chi_id chi_ex
        # 1/chi = 1/(chi_ex * chi_id)
        chi = chi_ex * chi_id
        chi_r = chi_id_r / chi_ex
    # end if (syst['dens']['totnum'] == 0.0)

    print("\tisothermal compressibility (using sr-c(r))")
    print("\t\texcess chi, chi^(-1)\t%f    %f" % (chi_ex, chi_ex_r))
    print("\t\tideal chi, chi^(-1)\t%.5e %f" % (chi_id, chi_id_r))
    print("\t\tabsolute, chi, chi^(-1)\t%.5e %f" % (chi, chi_r))
    print("")
