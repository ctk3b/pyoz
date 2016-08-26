import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz

plt.style.use('seaborn-colorblind')

T = 200
sig1 = 4.0
eps1 = 100 / T
sig2 = 6.0
eps2 = 25 / T
rho1 = 0.3 / sig1**3
rho2 = 0.3 / sig2**3


# Initialize a blank system and a Lennard-Jones potential with mixing rules.
lj_coul_binary = oz.System(T=T)
lj = oz.LennardJones(system=lj_coul_binary, sig='arithmetic', eps='geometric')
coul = oz.Coulomb(system=lj_coul_binary)

# Create and add component `M` to the system.
m = oz.Component(name='M', concentration=rho1)
m.add_potential(lj, sig=sig1, eps=eps1)
m.add_potential(coul, q=0.5)
lj_coul_binary.add_component(m)

# Create and add component `N` to the system.
n = oz.Component(name='N', concentration=rho2)
n.add_potential(lj, sig=sig2, eps=eps2)
n.add_potential(coul, q=0.5)
lj_coul_binary.add_component(n)

lj_coul_binary.solve(closure='hnc')


# Extract some results.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
max_r = 500
r, g_r, U_r = lj_coul_binary.r, lj_coul_binary.g_r, lj_coul_binary.U_r
for i, j in np.ndindex(lj_coul_binary.n_components, lj_coul_binary.n_components):
    if j < i:
        continue
    ax1.plot(r[:max_r], g_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
    ax2.plot(r[:max_r], U_r.ij[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='lower right')
fig1.savefig('g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('U(r) (kT)')
ax2.set_ylim((-0.20, 2.))
ax2.set_xlim((2, 12))
ax2.legend(loc='upper right')
fig2.savefig('U_r.pdf', bbox_inches='tight')
