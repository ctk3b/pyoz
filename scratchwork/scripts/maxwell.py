import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import seaborn as sns

sns.set_style('whitegrid')


T_range = np.arange(0.7, 1.3, 0.05)[5:]
d_rho = 0.010
rho_vap_min = 0.01
rho_vap_max = 0.2
rho_liq_min = 0.3
rho_liq_max = 0.8
rhos = np.append(np.arange(rho_vap_min, rho_vap_max, d_rho),
                      np.arange(rho_liq_min, rho_liq_max, d_rho))

sigma = 1

pressures = np.load('pressures_HNC.npy')[5:] *sigma**3
chem_pots = np.load('chem_pots_HNC.npy')[5:]
B2 =        np.load('vir_coeff_HNC.npy')[5:]
#converged = np.load('RHNC_converged.npy')[5:]

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def residual(rho_pair, mu_v_fit, mu_l_fit, P_v_fit, P_l_fit, weight_mu=0.5):
    weight_P = 1 - weight_mu
    rho_v, rho_l = rho_pair
    d_mu = np.polyval(mu_v_fit, rho_v) - np.polyval(mu_l_fit, rho_l)
    d_P = np.polyval(P_v_fit, rho_v) - np.polyval(P_l_fit, rho_l)
    return np.abs(d_mu) * weight_mu + np.abs(d_P) * weight_P


def plot_fit(ax, x, y, point, fit):
    ax.plot(x, y, marker='o', ms=4, color='b')
    ax.plot(x, np.polyval(fit, x), color='r')
    ax.plot(point, np.polyval(fit, point), marker='o', ms=10, color='k')


def calc_rho_pair(T_idx):
    vap_end = find_nearest(rhos, rho_vap_max)
    rho_v = rhos[:vap_end]
    liq_beg = find_nearest(rhos, rho_liq_min)
    rho_l = rhos[liq_beg:]

    # Chem pot data
    mu_ex = chem_pots[T_idx]
    mu_ex_v = mu_ex[:vap_end]
    mu_ex_l = mu_ex[liq_beg:]

    idx = np.isfinite(mu_ex_v)
    mu_rho_v = rho_v[idx]
    mu_ex_v = mu_ex_v[idx]
    mu_v = mu_ex_v + np.log(mu_rho_v)

    idx = np.isfinite(mu_ex_l)
    mu_rho_l = rho_l[idx]
    mu_ex_l = mu_ex_l[idx]
    mu_l = mu_ex_l + np.log(mu_rho_l)

    # Pressure data
    P = pressures[T_idx]
    P_v = P[:vap_end]
    P_l = P[liq_beg:]

    idx = np.isfinite(P_v)
    P_rho_v = rho_v[idx]
    P_v = P_v[idx]

    idx = np.isfinite(P_l)
    P_rho_l = rho_l[idx]
    P_l = P_l[idx]

    # Fit
    P_v_fit = np.polyfit(P_rho_v, P_v, 3)
    P_l_fit = np.polyfit(P_rho_l, P_l, 3)
    mu_v_fit = np.polyfit(mu_rho_v, mu_v, 3)
    mu_l_fit = np.polyfit(mu_rho_l, mu_l, 3)

    # Optimize
    vap_upper_bound = min(mu_rho_v[-1], P_rho_v[-1])
    liq_lower_bound = max(mu_rho_l[0], P_rho_l[0])
    print(vap_upper_bound, liq_lower_bound)
    data = least_squares(residual, (vap_upper_bound, liq_lower_bound),
                         bounds=([0, liq_lower_bound], [vap_upper_bound, 1.0]),
                         args=(mu_v_fit, mu_l_fit, P_v_fit, P_l_fit, 0.5))

    # Plot
    fig, ax = plt.subplots()
    plot_fit(ax, P_rho_v, P_v, data.x[0], P_v_fit)
    plot_fit(ax, P_rho_l, P_l, data.x[1], P_l_fit)
    ax.set_ylabel(r'$P^*$')
    ax.set_xlabel(r'$\rho^*$')
    fig.savefig('HNC_T{:.2f}_P_rho_pair.pdf'.format(T_range[T_idx]))
    fig.clear()

    fig, ax = plt.subplots()
    plot_fit(ax, mu_rho_v, mu_v, data.x[0], mu_v_fit)
    plot_fit(ax, mu_rho_l, mu_l, data.x[1], mu_l_fit)
    ax.set_ylabel((r'$\mu$'))
    ax.set_xlabel(r'$\rho^*$')
    fig.savefig('HNC_T{:.2f}_mu_rho_pair.pdf'.format(T_range[T_idx]))
    fig.clear()
    return data.x


# calc_rho_pair(6)
T_rhos = dict()
vap = []
liq = []
for T_idx, T in enumerate(T_range):
    v, l = calc_rho_pair(T_idx)
    vap.append(v)
    liq.append(l)

fig, ax = plt.subplots()
ax.plot(vap, T_range, color='r', label='maxwell construction')
ax.plot(liq, T_range, color='r')




T = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
liq = [0.71767145,
      0.65602262,
      0.59240108,
      0.51942802,
      0.40565828,
      0.24091243]

vap = [0.03783854,
       0.05178379,
       0.07833387,
       0.10350907,
       0.13553001,
       0.21122284]

ax.plot(liq, T, color='b', label=r'spinodal: fit S(k=0)')
ax.plot(vap, T, color='b')


r_v = [0.02, 0.03, 0.06, 0.1, 0.18]
T = [0.9, 1.0, 1.1, 1.2, 1.3]
r_l = [0.74, 0.68, 0.62, 0.55, 0.45]
ax.plot(r_v, T, color='k', label='binodal (literature)')
ax.plot(r_l, T, color='k')

ax.set_xlabel(r'$\rho^*$')
ax.set_ylabel(r'$T^*$')

ax.set_xlim((0, 0.8))
ax.set_ylim((0.7, 1.4))
ax.legend(loc='lower center')
fig.savefig('HNC_phase_maxwell.pdf')
