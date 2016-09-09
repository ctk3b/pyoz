import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import pyoz as oz
from pyoz.exceptions import PyozError

plt.style.use('seaborn-colorblind')

# Plot just the LJ potential.
fig_gr, ax_gr = plt.subplots()
fig_ur, ax_ur = plt.subplots()
fig_sk, ax_sk = plt.subplots()
sig = 1
eps = 1

d_rho = 0.01
rho_vap_min = 0.01
rho_vap_max = 0.2
rho_liq_min = 0.4
rho_liq_max = 0.8

rho_vaps = np.arange(rho_vap_min, rho_vap_max, d_rho)
rho_liqs = np.arange(rho_liq_min, rho_liq_max, d_rho)
rhos = np.append(rho_vaps, rho_liqs) / sig**3

Ts = np.arange(0.8, 1.41, .1)


properties = ['B2', 'P_virial', 'mu', 'Sk0', 's2']
data = np.empty(shape=(len(Ts), len(rhos), len(properties)))
data[:] = np.nan
thermo_data = xr.DataArray(data=data,
                           dims=('T', 'rho', 'data'),
                           coords={'T': Ts,
                                   'rho': rhos,
                                   'data': properties})

unary = oz.System()
properties = ['g_r', 'U_r', 'S_k']
data = np.empty(shape=(len(Ts), len(rhos), len(properties), len(unary.r)))
data[:] = np.nan
raw_data = xr.DataArray(data=data,
                        dims=('T', 'rho', 'data', 'r'),
                        coords={'T': Ts,
                                'rho': rhos,
                                'data': properties,
                                'r': unary.r})

G_r = None
for n, (T, rho) in enumerate(it.product(Ts, rhos)):
    oz.logger.info('T={:.3f}, rho={:.3f}'.format(T, rho))
    unary = oz.System(T=T)

    # Solve for the reference state.
    unary.set_interaction(0, 0, oz.wca(unary.r, eps=eps / T, sig=sig, m=12, n=6))
    try:
        g_r, c_r, G_r, S_k = unary.solve(
            rhos=rho, closure_name='hnc', mix_param=0.8, status_updates=False,
            initial_G_r=G_r)
    except PyozError as e:
        oz.logger.info(e)
        continue

    # Plot reference state.
    r, k, U_r = unary.r, unary.k, unary.U_r
    label = 'WCA' if n == 0 else ''
    ax_gr.plot(r, g_r[0, 0], lw=1.5, label=label)
    ax_ur.plot(r, U_r[0, 0], lw=1.5, label=label)
    ax_sk.plot(k, S_k[0, 0], lw=1.5, label=label)

    # Solve for the actual state.
    unary = oz.System(T=T)
    unary.set_interaction(0, 0, oz.lennard_jones(unary.r, eps=eps / T, sig=sig))
    try:
        g_r, c_r, G_r, S_k = unary.solve(
            rhos=rho, closure_name='rhnc', mix_param=0.8, g_r_ref=g_r,
            G_r_ref=G_r, U_r_ref=U_r, status_updates=False,
            initial_G_r=G_r)
    except PyozError as e:
        oz.logger.info(e)
        continue

    # Plot actual state.
    r, k, U_r = unary.r, unary.k, unary.U_r
    label = 'LJ' if n == 0 else ''
    ax_gr.plot(r, g_r[0, 0], lw=1.5, label=label)
    ax_ur.plot(r, U_r[0, 0], lw=1.5, label=label)
    ax_sk.plot(k, S_k[0, 0], lw=1.5, label=label)

    raw_data.loc[T, rho, 'g_r'] = g_r[0, 0]
    raw_data.loc[T, rho, 'U_r'] = U_r[0, 0]
    raw_data.loc[T, rho, 'S_k'] = S_k[0, 0]

    B2 = oz.second_virial_coefficient(unary)
    P_virial = oz.pressure_virial(unary)
    mu_ex = T * oz.excess_chemical_potential(unary)[0]
    mu = mu_ex + T * np.log(rho)
    s2 = oz.two_particle_excess_entropy(unary)

    thermo_data.loc[T, rho] = [B2, P_virial, mu, S_k[0, 0, 0], s2]


ds = raw_data.to_dataset(name='raw')
ds.to_netcdf('data/RHNC_raw_data.nc')

ds = thermo_data.to_dataset(name='thermo')
ds.to_netcdf('data/RHNC_thermo_data.nc')

ax_gr.set_xlabel('r (Å)')
ax_gr.set_ylabel('g(r)')
ax_gr.set_xlim((0, 5))
ax_gr.legend(loc='upper right')
fig_gr.savefig('figs/g_r.pdf', bbox_inches='tight')

ax_ur.set_xlabel('r (Å)')
ax_ur.set_ylabel('U(r) (kT)')
ax_ur.set_ylim((-1.5, 2.))
ax_ur.set_xlim((0, 12))
ax_ur.legend(loc='upper right')
fig_ur.savefig('figs/U_r.pdf', bbox_inches='tight')

ax_sk.set_xlabel('k')
ax_sk.set_ylabel('S(k)')
ax_sk.legend(loc='upper right')
ax_sk.set_xlim((0, 50))
fig_sk.savefig('figs/S_k.pdf', bbox_inches='tight')

