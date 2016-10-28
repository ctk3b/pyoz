import matplotlib.pyplot as plt
import numpy as np

import pyoz as oz

plt.style.use('seaborn-colorblind')

# Initialize a blank system and a DPD potential.
dpd_binary = oz.System()


def dpd_func(r, a):
    cutoff = np.abs(r - 1.0).argmin()
    dpd = np.zeros_like(r)
    dpd[:cutoff] = 0.5 * a * (1 - r[:cutoff])**2
    return dpd

dpd_binary.set_interaction(0, 0, dpd_func(dpd_binary.r, 15))
dpd_binary.set_interaction(0, 1, dpd_func(dpd_binary.r, 17))
dpd_binary.set_interaction(1, 1, dpd_func(dpd_binary.r, 15))
g_r, _, _, _ = dpd_binary.solve(rhos=[2, 1], closure_name='hnc')


# Extract some results.
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
max_r = 1000
r, U_r = dpd_binary.r, dpd_binary.U_r
for i, j in np.ndindex(dpd_binary.n_components, dpd_binary.n_components):
    if j < i:
        continue
    ax1.plot(r[:max_r], g_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))
    ax2.plot(r[:max_r], U_r[i, j, :max_r], lw=1.5, label='{}{}'.format(i, j))

ax1.set_xlabel('r (Å)')
ax1.set_ylabel('g(r)')
ax1.legend(loc='lower right')
ax1.set_xlim((0, 5))
fig1.savefig('g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('U(r) (kT)')
ax2.legend(loc='lower right')
ax2.set_xlim((0, 5))
fig2.savefig('U_r.pdf', bbox_inches='tight')
