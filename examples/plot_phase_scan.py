import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

sns.set_style('whitegrid')
colors = sns.color_palette()


thermo_data = xr.open_dataset('data/RHNC_thermo_data.nc')['thermo']
# For some reason the data-type isn't preserved and loads back as '|S8'
thermo_data.coords['data'] = thermo_data.coords['data'].astype('<U8')
raw_data = xr.open_dataset('data/RHNC_raw_data.nc')['raw']
raw_data.coords['data'] = raw_data.coords['data'].astype('<U8')

# TODO: Subsume into data
sig = 1
eps = 1

rho_vap_min = 0.01
rho_vap_max = 0.2
rho_liq_min = 0.4
rho_liq_max = 0.8

# CONVERGENCE
# fig = plt.figure()
# ax = sns.heatmap(converged, xticklabels=rhos, yticklabels=T_range, cbar=False)
# valid = [str(x) for x in np.arange(0, 0.9, 0.1)]
# for label in ax.xaxis.get_ticklabels():
#     if label.get_text() not in valid:
#         label.set_visible(False)
#
# valid = [str(x) for x in np.arange(0.7, 1.4, 0.1)]
# for label in ax.yaxis.get_ticklabels():
#     if label.get_text() not in valid:
#         label.set_visible(False)
#
#
# ax.invert_yaxis()
# ax.set_xlabel(r'$\rho^*$')
# ax.set_ylabel(r'$T^*$')
# fig.savefig('RHNC_phase.pdf', bbox_inches='tight')


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

# PRESSURE
fig_v, ax_v = plt.subplots()
fig_l, ax_l = plt.subplots()
temps = thermo_data['T']
rhos = thermo_data['rho']
rho_vap_max = 0.2

for T, color in zip(temps, it.cycle(colors)):
    rho_vmax_idx = find_nearest(rhos, rho_vap_max).values
    rho_v = rhos[:rho_vmax_idx]
    T_form = '{:.2f}'.format(float(T))

    pressures = thermo_data.loc[T, :, 'P_virial'] * sig**3
    B2 = thermo_data.loc[T, :, 'B2'][0] / sig**3

    ax_v.plot(rho_v, pressures[:rho_vmax_idx],
              marker='o', lw=0, ms=3, label=T_form, color=color)
    ax_v.plot(rho_v, (rho_v + B2 * rho_v**2) * T,
              color=color)

    ax_l.plot(rhos, pressures,
              marker='o', lw=0, ms=3, label=T_form, color=color)
ax_v.set_xlim((0, rho_vap_max))
ax_v.set_ylim((0, 0.07))
ax_l.set_xlim((rho_liq_min, rho_liq_max))


ax_v.set_xlabel(r'$\rho^*$')
ax_l.set_xlabel(r'$\rho^*$')
ax_v.set_ylabel(r'P')
ax_l.set_ylabel(r'P')
ax_v.legend(loc='upper left')
ax_l.legend(loc='upper left')
fig_v.savefig('figs/RHNC_P_vap.pdf', bbox_inches='tight')
fig_l.savefig('figs/RHNC_P_liq.pdf', bbox_inches='tight')
#
# #  CHEM POT
# fig_v, ax_v = plt.subplots()
# fig_l, ax_l = plt.subplots()
# for i, (T, color) in enumerate(zip(T_range, it.cycle(colors))):
#     rho_vmax_idx = find_nearest(rhos, rho_vap_max)
#     rho_v = rhos[:rho_vmax_idx]
#     T_form = '{:.2f}'.format(T)
#
#     ax_v.plot(rho_v, chem_pots[i, :rho_vmax_idx],
#               marker='o', lw=0, ms=3, label=T_form, color=color)
#     ax_v.plot(rho_v, T * np.log(rho_v),
#               color=color)
#     ax_l.plot(rhos, chem_pots[i, :],
#               marker='o', lw=0, ms=3, label=T_form, color=color)
# ax_v.set_xlim((0, rho_vap_max))
# ax_l.set_xlim((rho_liq_min, rho_liq_max))
#
#
# ax_v.set_xlabel(r'$\rho^*$')
# ax_l.set_xlabel(r'$\rho^*$')
# ax_v.set_ylabel(r'$\mu^{ex}$')
# ax_l.set_ylabel(r'$\mu^{ex}$')
# ax_v.legend(loc='lower right')
# ax_l.legend(loc='upper left')
# fig_v.savefig('RHNC_mu_vap.pdf', bbox_inches='tight')
# fig_l.savefig('RHNC_mu_liq.pdf', bbox_inches='tight')
