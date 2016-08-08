import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simtk.unit as u

from pyoz import solve_ornstein_zernike

sns.set_style('whitegrid')


inputs = dict()
inputs['n_points'] = 4096  # use power of 2!
inputs['dr'] = 0.05 * u.angstrom
inputs['mix_param'] = 1.0
inputs['tol'] = 1e-9
inputs['max_iter'] = 5000

# system information
inputs['T'] = 298.15 * u.kelvin
n_components = 2
inputs['n_components'] = n_components
inputs['closure'] = 'hnc'
inputs['names'] = ['P', 'M']

inputs['potentials'] = dict()

lj = inputs['potentials']['lennard-jones'] = dict()
lj['sigmas'] = [0.4 * u.nanometers,
                0.6 * u.nanometers]
lj['sigma_rule'] = 'arithmetic'
lj['epsilons'] = [1.0 * u.kilojoules_per_mole,
                  2.0 * u.kilojoules_per_mole]
lj['epsilon_rule'] = 'geometric'


max_r = 200
for mol_L in [1]:
    print('Concentration: {} mol/L'.format(mol_L))
    inputs['concentrations'] = [mol_L * u.moles / u.liter,
                                mol_L * u.moles / u.liter]
    r, g_r_ij = solve_ornstein_zernike(inputs)
    for i, j in it.product(range(n_components), range(n_components)):
        plt.plot(r[:max_r], g_r_ij[i, j, :max_r], label='{}{}'.format(i, j))

plt.legend(loc='upper left')
plt.savefig('g_r.pdf', bbox_inches='tight')
