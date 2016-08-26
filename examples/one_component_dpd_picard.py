import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz

plt.style.use('seaborn-colorblind')

# Initialize a blank system and a DPD potential.
dpd_unary = oz.System()


def dpd_func(r, a):
    cutoff = np.abs(r - 1.0).argmin()
    dpd = np.zeros_like(r)
    dpd[:cutoff] = 0.5 * a * (1 - r[:cutoff])**2
    return dpd

potential = oz.ContinuousPotential(system=dpd_unary,
                                   potential_func=dpd_func,
                                   a='arithmetic')

# Create and add component `M` to the system.
m = oz.Component(name='M', concentration=5)
m.add_potential(potential, a=50)
dpd_unary.add_component(m)

dpd_unary.solve(closure='hnc')


# Extract some results.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
max_r = 100
r, g_r, U_r = dpd_unary.r, dpd_unary.g_r, dpd_unary.U_r

ax1.plot(r[:max_r], g_r[0, 0, :max_r], lw=1.5)
ax2.plot(r[:max_r], U_r.ij[0, 0, :max_r], lw=1.5)

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
fig1.savefig('g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('g(r)')
fig2.savefig('U_r.pdf', bbox_inches='tight')

kb = oz.kirkwood_buff_integrals(dpd_unary)
print('Kirkwood-Buff integrals:\n', kb)
mu_ex = oz.excess_chemical_potential(dpd_unary)
print('Excess chemical potentialls:\n', mu_ex)

