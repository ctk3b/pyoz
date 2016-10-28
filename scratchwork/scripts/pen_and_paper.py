import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import pyoz as oz

sns.set_style('whitegrid', {'xtick.major.size': 5,
                            'xtick.labelsize': 'large',
                            'ytick.major.size': 5,
                            'ytick.labelsize': 'large',
                            'axes.edgecolor': 'k',
                            'font.weight': 'bold',
                            'axes.labelsize': 'large',
})
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.style.use('seaborn-colorblind')




rhos = np.arange(0, 2, 0.2)
rho_ds = np.arange(0.1, 2, 0.2)

sig_c = 2.2
sig_d = 1.0

converged = pd.DataFrame(index={'rho': rhos}, columns={'rho_d': rho_ds})

for rho, rho_d in it.product(rhos, rho_ds):
    soft = oz.System()
    U = oz.soft_depletion(soft.r, eps=1, n=36, sig_c=sig_c, sig_d=sig_d, rho_d=rho_d)
    soft.set_interaction(0, 0, U)

    g_r, _, _, _ = soft.solve(rhos=rho, closure_name='hnc')

    if np.isnan(g_r).all():
        converged.set_value(rho, rho_d, False)
    else
        converged.set_value(rho, rho_d, True)



