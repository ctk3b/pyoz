import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz
from pyoz.exceptions import PyozError

plt.style.use('seaborn-colorblind')

T = 119.8
sig1 = 3.405
sig2 = 3.405
eps1 = 119.8 / T
eps2 = 119.8 / T


# for rho in np.arange(0.55, 0.8, 0.001):
for rho in [0.6]:
    rho1 = rho / sig1**3
    rho2 = rho / sig2**3
    # Initialize a blank system and a Lennard-Jones potential with mixing rules.
    lj_coul_binary = oz.System(T=T)
    lj = oz.LennardJones(system=lj_coul_binary, sig='arithmetic', eps='geometric')
    # coul = oz.Coulomb(system=lj_coul_binary)

    # Create and add component `M` to the system.
    m = oz.Component(name='M', rho=rho1)
    m.add_potential(lj, sig=sig1, eps=eps1)
    # m.add_potential(coul, q=0.0)
    lj_coul_binary.add_component(m)

    # Create and add component `N` to the system.
    n = oz.Component(name='N', rho=rho2)
    n.add_potential(lj, sig=sig2, eps=eps2)
    # n.add_potential(coul, q=0.0)
    lj_coul_binary.add_component(n)

    try:
        lj_coul_binary.solve(closure='hnc', mix_param=0.8, status_updates=False)
    except PyozError as e:
        oz.logger.info(e)


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
max_r = 500
r, g_r, U_r, k, S_k = lj_coul_binary.r, lj_coul_binary.g_r, lj_coul_binary.U_r, lj_coul_binary.k, lj_coul_binary.S_k
for i, j in np.ndindex(lj_coul_binary.n_components, lj_coul_binary.n_components):
    if j < i:
        continue
    ax1.plot(r[:max_r], g_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
    ax2.plot(r[:max_r], U_r.ij[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
    ax3.plot(k[:max_r], S_k[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='lower right')
ax1.set_xlim((0, 15))
fig1.savefig('g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('U(r) (kT)')
ax2.legend(loc='upper right')
ax2.set_ylim((-2.00, 1.00))
ax2.set_xlim((0, 15))
fig2.savefig('U_r.pdf', bbox_inches='tight')

ax3.set_xlabel('k')
ax3.set_ylabel('S(k)')
ax3.legend(loc='lower right')
ax3.set_xlim((0, 10))
fig3.savefig('S_k.pdf', bbox_inches='tight')

kb = oz.kirkwood_buff_integrals(lj_coul_binary)
print('Kirkwood-Buff integrals:\n', kb)
mu_ex = oz.excess_chemical_potential(lj_coul_binary)
print('Excess chemical potentialls:\n', mu_ex)