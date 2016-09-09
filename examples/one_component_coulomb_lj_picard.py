import matplotlib.pyplot as plt

import pyoz as oz

plt.style.use('seaborn-colorblind')

# WARNING: This system is positively charged and hence unphysical.

T = 119.8
sigma = 3.405
epsilon = 119.8 / T
rho = 0.60 / sigma**3

# Initialize a blank system and a Lennard-Jones potential with mixing rules.
unary = oz.System(T=T)
lj = oz.LennardJones(system=unary, sig='arithmetic', eps='geometric')
m = oz.Component(name='M', rho=rho)
m.add_potential(lj, sig=sigma, eps=epsilon)
unary.add_component(m)
unary.solve(closure_name='hnc', mix_param=0.8)


# Plot just the LJ potential.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
max_r = 500
r, g_r, U_r, k, S_k = unary.r, unary.g_r, unary.U_r, unary.k, unary.S_k
ax1.plot(r[:max_r], g_r[0, 0, :max_r], lw=1.5, label='LJ only')
ax2.plot(r[:max_r], U_r.ij[0, 0, :max_r], lw=1.5, label='LJ only')
ax3.plot(k[:max_r], S_k[0, 0, :max_r], ls='-', lw=2.0, label='LJ only')


# Add a coulomb potential and re-solve the system.
coul = oz.Coulomb(system=unary)
m.add_potential(coul, q=.05)
# unary.solve(closure='hnc', max_iter=2000)


# Plot both potentials.
r, g_r, U_r, k, S_k = unary.r, unary.g_r, unary.U_r, unary.k, unary.S_k
ax1.plot(r[:max_r], g_r[0, 0, :max_r], lw=1.5, label='LJ + Coul')
ax2.plot(r[:max_r], U_r.ij[0, 0, :max_r], lw=1.5, label='LJ + Coul')
ax3.plot(k[:max_r], S_k[0, 0, :max_r], ls=':', lw=1.5, label='LJ + Coul')

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='lower right')
fig1.savefig('g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('U(r) (kT)')
ax2.set_ylim((-1.5, 2.))
ax2.set_xlim((2, 12))
ax2.legend(loc='upper right')
fig2.savefig('U_r.pdf', bbox_inches='tight')

ax3.set_xlabel('k')
ax3.set_ylabel('S(k)')
ax3.legend(loc='lower right')
fig3.savefig('S_k.pdf', bbox_inches='tight')

kb = oz.kirkwood_buff_integrals(unary)
print('Kirkwood-Buff integrals:\n', kb)
mu_ex = oz.excess_chemical_potential(unary)
print('Excess chemical potentialls:\n', mu_ex)