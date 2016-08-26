import matplotlib.pyplot as plt

import pyoz as oz

plt.style.use('seaborn-colorblind')

T = 119.8
sigma = 3.405
epsilon = 119.8 / T
rho = 0.01 / sigma**3

# Initialize a blank system and a Lennard-Jones potential with mixing rules.
unary = oz.System()
lj = oz.LennardJones(system=unary, sig='arithmetic', eps='geometric')
m = oz.Component(name='M', concentration=rho)
m.add_potential(lj, sig=sigma, eps=epsilon)
unary.add_component(m)
unary.solve(closure='hnc')


# Plot just the LJ potential.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
max_r = 500
r, g_r, U_r = unary.r, unary.g_r, unary.U_r
ax1.plot(r[:max_r], g_r[0, 0, :max_r], lw=1.5, label='LJ only')
ax2.plot(r[:max_r], U_r.ij[0, 0, :max_r], lw=1.5, label='LJ only')


# Add a coulomb potential and re-solve the system.
coul = oz.Coulomb(system=unary)
m.add_potential(coul, q=.5)
unary.solve(closure='hnc')


# Plot both potentials.
r, g_r, U_r = unary.r, unary.g_r, unary.U_r
ax1.plot(r[:max_r], g_r[0, 0, :max_r], lw=1.5, label='LJ + Coul')
ax2.plot(r[:max_r], U_r.ij[0, 0, :max_r], lw=1.5, label='LJ + Coul')

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
