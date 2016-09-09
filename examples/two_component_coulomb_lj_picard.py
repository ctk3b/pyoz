import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz
from pyoz.exceptions import PyozError

plt.style.use('seaborn-colorblind')


T = 1

sig0 = 1
eps0 = 1 / T
sig1 = 2
eps1 = 1 / T
sig01 = oz.arithmetic(sig1, sig0)
eps01 = oz.geometric(sig1, sig0)

q0 = -0.8
q1 = 0.8

rho1 = 0.01 / sig0**3
rho2 = 0.01 / sig1**3


# Initialize a blank system and a Lennard-Jones potential with mixing rules.
lj_coul_binary = oz.System(T=T, n_points=8192)
r = lj_coul_binary.r
lj_coul_binary.set_interaction(0, 0,
                               oz.lennard_jones(r, sig=sig0, eps=eps0) +
                               oz.coulomb(r, q0, q0))
lj_coul_binary.set_interaction(0, 1,
                               oz.lennard_jones(r, sig=sig01, eps=eps01) +
                               oz.coulomb(r, q0, q1))
lj_coul_binary.set_interaction(1, 1,
                               oz.lennard_jones(r, sig=sig1, eps=eps1) +
                               oz.coulomb(r, q1, q1))

g_r, _, _, S_k = lj_coul_binary.solve(rhos=[rho1, rho2])


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
r, g_r, U_r, k, S_k = lj_coul_binary.r, lj_coul_binary.g_r, lj_coul_binary.U_r, lj_coul_binary.k, lj_coul_binary.S_k
for i, j in np.ndindex(lj_coul_binary.n_components, lj_coul_binary.n_components):
    if j < i:
        continue
    ax1.plot(r, g_r[i, j], lw=1.5, label='{}{}'.format(i, j))
    ax2.plot(r, U_r[i, j], lw=1.5, label='{}{}'.format(i, j))
    ax3.plot(k, S_k[i, j], lw=1.5, label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='upper right')
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
