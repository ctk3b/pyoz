import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz
from pyoz.exceptions import PyozError
import pyoz.unit as u

plt.style.use('seaborn-colorblind')
fig_gr, ax_gr = plt.subplots()
fig_ur, ax_ur = plt.subplots()
max_r = 500



#sigma = 3.405
sigma = 1

d_rho = 0.010
rho_vap_min = 0.01
rho_vap_max = 0.2
rho_liq_min = 0.4
rho_liq_max = 0.8
rho_range = np.append(np.arange(rho_vap_min, rho_vap_max, d_rho),
                      np.arange(rho_liq_min, rho_liq_max, d_rho))
rho_red_range = rho_range / sigma**3
T_range = np.arange(1.0, 1.41, 5.1)
# T_range = [1.0]

epsilon = 1


pressures = np.ones(shape=(len(T_range), len(rho_red_range)))
chem_pots = np.ones(shape=(len(T_range), len(rho_red_range)))
vir_coeff = np.ones(shape=(len(T_range), len(rho_red_range)))
converged = np.ones(shape=(len(T_range), len(rho_red_range)))

previous_system = None
for i, T_red in enumerate(T_range):
    T_real = T_red * epsilon
    for j, rho_red in enumerate(rho_red_range):
        oz.logger.info('T*={:.2f}, rho_ij*={:.2f}'.format(T_red, rho_red*sigma**3))
        unary = oz.System(mix_param=0.8, T=T_real)
        lj = oz.LennardJones(unary)
        m = oz.Component(name='M', rho=rho_red)
        m.add_potential(lj, sig=sigma, eps=epsilon / T_real)
        unary.T_red = T_real / epsilon
        unary.add_component(m)

        if previous_system:
            guess_e_r = previous_system.e_r
        try:
            # unary.solve(closure_name='hnc', status_updates=False, max_iter=1500)
            unary.solve(status_updates=False, max_iter=1500)
        except PyozError as e:
            oz.logger.info(e)
            converged[i, j] = 0
            pressures[i, j] = None
            chem_pots[i, j] = None
            vir_coeff[i, j] = None
            continue

        r, g_r, U_r, S_k = unary.r, unary.g_r, unary.U_r, unary.S_k
        P = oz.pressure_virial(unary)
        mu = T_red * oz.excess_chemical_potential(unary)
        # B2 = oz.second_virial_coefficient(unary)

        pressures[i, j] = P
        print(P)
        chem_pots[i, j] = mu
        # vir_coeff[i, j] = B2
        ax_gr.plot(r[:max_r], g_r[0, 0, :max_r], lw=0.5, label='{:.2f} {:.2f}'.format(T_red, rho_red))
        ax_ur.plot(r[:max_r], U_r.ij[0, 0, :max_r], lw=0.5, label='{:.2f} {:.2f}'.format(T_red, rho_red))
        previous_system = unary

ax_gr.set_xlabel('r (Å)')
ax_gr.set_ylabel('g(r)')
# ax_gr.legend(loc='upper right')
fig_gr.savefig('g_r.pdf', bbox_inches='tight')

ax_ur.set_xlabel('r (Å)')
ax_ur.set_ylabel('U(r)')
# ax_ur.legend(loc='upper right')
ax_ur.set_ylim((-3, 2.))
ax_ur.set_xlim((2, 12))
fig_ur.savefig('u_r.pdf', bbox_inches='tight')

np.save('pressures_HNC.npy', pressures)
np.save('chem_pots_HNC.npy', chem_pots)
np.save('vir_coeff_HNC.npy', vir_coeff)
np.save('converged_HNC.npy', converged)

