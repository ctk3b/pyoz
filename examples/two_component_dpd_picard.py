import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz
import pyoz.unit as u

plt.style.use('seaborn-colorblind')

# Initialize a blank system and a DPD potential.
dpd_binary = oz.System()


def dpd_func(r, a):
    cutoff = np.abs(r - 1.0).argmin()
    dpd = np.zeros_like(r)
    dpd[:cutoff] = 0.5 * a * (1 - r[:cutoff])**2
    return dpd

potential = oz.ContinuousPotential(dpd_func, a_rule='arithmetic')

# Create and add component `M` to the system.
m = oz.Component(name='M', concentration=20000 / 6.022 * u.moles / u.liter)
m.add_potential(potential, parameters={'a': 37.5 * u.kilojoules_per_mole})
dpd_binary.add_component(m)

# Create and add component `N` to the system.
n = oz.Component(name='N', concentration=30000 / 6.022 * u.moles / u.liter)
n.add_potential(potential, parameters={'a': 37.5 * u.kilojoules_per_mole})
dpd_binary.add_component(n)

dpd_binary.solve(closure='hnc')


# Extract some results.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
max_r = 100
r, g_r, U_r = dpd_binary.r, dpd_binary.g_r, dpd_binary.U_r
for i, j in np.ndindex(dpd_binary.n_components, dpd_binary.n_components):
    if j < i:
        continue
    ax1.plot(r[:max_r], g_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
    ax2.plot(r[:max_r], U_r.ij[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='lower right')
fig1.savefig('g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('g(r)')
ax2.legend(loc='lower right')
fig2.savefig('U_r.pdf', bbox_inches='tight')

kb = oz.kirkwood_buff_integrals(dpd_binary)
print('Kirkwood-Buff integrals:\n', kb)
mu_ex = oz.excess_chemical_potential(dpd_binary)
print('Excess chemical potentialls:\n', mu_ex)

