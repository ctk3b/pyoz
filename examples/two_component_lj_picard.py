import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz
from pyoz.potentials import arithmetic, geometric
from pyoz.exceptions import PyozError

plt.style.use('seaborn-colorblind')

T = 1

sig0 = 1
eps0 = 1 / T
sig1 = 2
eps1 = 1 / T
sig01 = arithmetic(sig1, sig0)
eps01 = geometric(sig1, sig0)

rho1 = 0.01 / sig0**3
rho2 = 0.01 / sig1**3


# Initialize a blank system and a Lennard-Jones potential with mixing rules.
lj_binary = oz.System(T=T, dr=0.01, n_points=8192)
r = lj_binary.r
lj_binary.set_interaction(0, 0, oz.lennard_jones(r, sig=sig0, eps=eps0))
lj_binary.set_interaction(0, 1, oz.lennard_jones(r, sig=sig01, eps=eps01))
lj_binary.set_interaction(1, 1, oz.lennard_jones(r, sig=sig1, eps=eps1))

g_r, _, _, S_k = lj_binary.solve(rhos=[rho1, rho2])

import ipdb; ipdb.set_trace()

# Extract some results.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
max_r = 1000
r, U_r, k = lj_binary.r, lj_binary.U_r, lj_binary.k
for i, j in np.ndindex(lj_binary.n_components, lj_binary.n_components):
    if j < i:
        continue
    ax1.plot(r[:max_r], g_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
    ax2.plot(r[:max_r], U_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
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
