import math
import glob
import pickle
import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


thermo_data = xr.open_dataset('data/RHNC_thermo_data.nc')['thermo']
# For some reason the data-type isn't preserved and loads back as '|S8'
thermo_data.coords['data'] = thermo_data.coords['data'].astype('<U8')
raw_data = xr.open_dataset('data/RHNC_raw_data.nc')['raw']
raw_data.coords['data'] = raw_data.coords['data'].astype('<U8')
with open('all_data.pkl', 'rb') as f:
    data = pickle.load(f)

fig_gr, ax_gr = plt.subplots()
fig_ur, ax_ur = plt.subplots()
fig_sk, ax_sk = plt.subplots()
fig_rho_P, ax_rho_P = plt.subplots()

prev_labels = set()
colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c', 'r']
all_files = glob.glob('scan/test_exp_eps*.pkl')
for n, file_name in enumerate(all_files):
    print(n, file_name)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    r = data['r']
    g_r = data['g_r']
    if all(np.isnan(x) for x in g_r):
        continue
    U_r = data['U_r']
    k = data['k']
    S_k = data['S_k']
    rho = data['rho_ij']
    eps = data['eps']
    P = data['P_virial']
    m = data['m']
    n = data['n']

    label = '{:.3f} {} {}'.format(eps, m, n)
    if label in prev_labels:
        label = None
    else:
        prev_labels.add(label)

    color = colors[int(m / 12 - 1)]

    ax_gr.plot(r, g_r, lw=0.1, label=label, color=color)
    ax_ur.plot(r, U_r, lw=0.1, label=label, color=color)
    ax_sk.plot(k, S_k, lw=0.1, label=label, color=color)

    ax_rho_P.plot(rho, P, marker='o', label=label, color=color)

ax_gr.set_xlabel('r')
ax_gr.set_ylabel('g(r)')
ax_gr.set_xlim((0, 5))
ax_gr.legend(loc='upper right')
fig_gr.savefig('figs/g_r.pdf', bbox_inches='tight')

ax_ur.set_xlabel('r')
ax_ur.set_ylabel('U(r)')
ax_ur.set_ylim(np.min(U_r)-0.1, np.mean(U_r[200:]) + 0.1)
ax_ur.set_xlim((0, 5))
ax_ur.legend(loc='upper right')
fig_ur.savefig('figs/U_r.pdf', bbox_inches='tight')

ax_sk.set_xlabel('k')
ax_sk.set_ylabel('S(k)')
ax_sk.legend(loc='upper right')
ax_sk.set_xlim((0, 50))
fig_sk.savefig('figs/S_k.pdf', bbox_inches='tight')


ax_rho_P.set_xlabel('rho_ij')
ax_rho_P.set_ylabel('P')
# ax_rho_P.legend(loc='upper right')
# ax_rho_P.set_xlim((0, 50))
fig_rho_P.savefig('figs/rho_P.pdf', bbox_inches='tight')
