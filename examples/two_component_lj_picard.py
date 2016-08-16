import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simtk.unit as u

import pyoz as oz

sns.set_style('whitegrid')


lj_liquid = oz.System()
potential = oz.LennardJones(sig_rule='arithmetic', eps_rule='geometric')

# Create and add component `M` to the system.
m = oz.Component(name='M', concentration=0.5 * u.moles / u.liter)
m.add_potential(potential, parameters={'sig': 0.4 * u.nanometers,
                                       'eps': 0.1 * u.kilojoules_per_mole})
lj_liquid.add_component(m)

# Create and add component `N` to the system.
n = oz.Component(name='N', concentration=0.5 * u.moles / u.liter)
n.add_potential(potential, parameters={'sig': 0.6 * u.nanometers,
                                       'eps': 0.1 * u.kilojoules_per_mole})
lj_liquid.add_component(n)


fig1, ax1 = plt.subplots()
n_components = lj_liquid.n_components
max_r = 500  # for plotting
for mol_L in [0.5]:
    m.concentration = n.concentration = mol_L * u.moles / u.liter
    lj_liquid.solve(closure='hnc')

    r, g_r = lj_liquid.r, lj_liquid.g_r
    for i, j in it.product(range(n_components), range(n_components)):
        ax1.plot(r[:max_r], g_r[i, j, :max_r], label='{}{}'.format(i, j))

    kb = oz.kirkwood_buff_integrals(lj_liquid)
    mu_ex = oz.excess_chemical_potential(lj_liquid)

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')

ax1.legend(loc='lower right')
fig1.savefig('g_r.pdf', bbox_inches='tight')
