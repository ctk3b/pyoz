import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simtk.unit as u

import pyoz as oz
from pyoz.potentials import LennardJones
from pyoz.core import System, Component

sns.set_style('whitegrid')


lj_liquid = System()
potential = LennardJones(sig_rule='arithmetic', eps_rule='geometric')

m = Component(name='M', concentration=0.5 * u.moles / u.liter)
m.add_potential(potential, parameters={'sig': 0.4 * u.nanometers,
                                       'eps': 0.1 * u.kilojoules_per_mole})
n = Component(name='N', concentration=0.5 * u.moles / u.liter)
n.add_potential(potential, parameters={'sig': 0.6 * u.nanometers,
                                       'eps': 0.1 * u.kilojoules_per_mole})

# TODO: remove need to add potential twice
lj_liquid.add_component(m)
lj_liquid.add_component(n)

max_r = 500
n_components = lj_liquid.n_components
fig1, ax1 = plt.subplots()
for mol_L in [0.5]:
    m.concentration = mol_L * u.moles / u.liter
    n.concentration = mol_L * u.moles / u.liter
    r, g_r = lj_liquid.solve(closure='hnc')
    kb = oz.compute_kirkwood_buff(r, g_r)
    print(kb)

    for i, j in it.product(range(n_components), range(n_components)):
        ax1.plot(r[:max_r], g_r[i, j, :max_r], label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')

ax1.legend(loc='lower right')
fig1.savefig('g_r.pdf', bbox_inches='tight')
