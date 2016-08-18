import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz
import pyoz.unit as u

plt.style.use('seaborn-pastel')


# Initialize a blank system and a Lennard-Jones potential with mixing rules.
lj_binary = oz.System()
potential = oz.LennardJones(sig_rule='arithmetic', eps_rule='geometric')

# Create and add component `M` to the system.
m = oz.Component(name='M', concentration=0.5 * u.moles / u.liter)
m.add_potential(potential, parameters={'sig': 0.4 * u.nanometers,
                                       'eps': 0.1 * u.kilojoules_per_mole})
lj_binary.add_component(m)

# Create and add component `N` to the system.
n = oz.Component(name='N', concentration=0.5 * u.moles / u.liter)
n.add_potential(potential, parameters={'sig': 0.6 * u.nanometers,
                                       'eps': 0.1 * u.kilojoules_per_mole})
lj_binary.add_component(n)

lj_binary.solve(closure='hnc')


fig1, ax1 = plt.subplots()
max_r = 500
r, g_r = lj_binary.r, lj_binary.g_r
for i, j in np.ndindex(lj_binary.n_components, lj_binary.n_components):
    if j < i:
        continue
    ax1.plot(r[:max_r], g_r[i, j, :max_r], label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='lower right')
fig1.savefig('g_r.pdf', bbox_inches='tight')

kb = oz.kirkwood_buff_integrals(lj_binary)
print('Kirkwood-Buff integrals:\n', kb)
mu_ex = oz.excess_chemical_potential(lj_binary)
print('Excess chemical potentialls:\n', mu_ex)

