import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from scipy.special import erf

import pyoz as oz
from pyoz.potentials import arithmetic, geometric
from pyoz.exceptions import PyozError

plt.style.use('seaborn-colorblind')
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()


T = 1

sig0 = 1
eps0 = 1 / T
sig1 = 1
eps1 = 1 / T
sig01 = arithmetic(sig1, sig0)
eps01 = geometric(sig1, sig0)

q0 = -0.5
q1 = 0.5

rho1 = 0.0001 / sig0**3
rho2 = 0.0001 / sig1**3

bje_l = 1
deb_l = 1 / np.sqrt(4 * np.pi * bje_l * (rho1 * (q0**2) + rho2 * (q1**2)))
kappa = 1 / deb_l
print(deb_l)

# Initialize a blank system and a Lennard-Jones potential with mixing rules.
lj_coul_binary = oz.System(kT=T, n_points=16384, dr=0.02)
r = lj_coul_binary.r
k = lj_coul_binary.k

coul_00 = oz.coulomb(r, q0, q0)
coul_01 = oz.coulomb(r, q0, q1)
coul_11 = oz.coulomb(r, q1, q1)

U_k_corr = exp(-1 * k**2 / (4 * 1.08**2)) / k**2
alpha = max(kappa, 0.05) / 2
U_r_corr = erf(alpha * r)

lj_coul_binary.set_interaction(
    0, 0,
    oz.wca(r, sig=sig0, eps=eps0, m=12, n=6) + coul_00,
    # oz.lennard_jones(r, sig=sig0, eps=eps0) + coul_00,
    long_range_real=coul_00 * U_r_corr,
    long_range_fourier=(coul_00 * r) * U_k_corr
)
lj_coul_binary.set_interaction(
    0, 1,
    oz.wca(r, sig=sig01, eps=eps01, m=12, n=6) + coul_01,
    # oz.lennard_jones(r, sig=sig01, eps=eps01) + coul_01,
    long_range_real=coul_01 * U_r_corr,
    long_range_fourier=(coul_01 * r) * U_k_corr
)
lj_coul_binary.set_interaction(
    1, 1,
    oz.wca(r, sig=sig1, eps=eps1, m=12, n=6) + coul_11,
    # oz.lennard_jones(r, sig=sig1, eps=eps1) + coul_11,
    long_range_real=coul_11 * U_r_corr,
    long_range_fourier=(coul_11 * r) * U_k_corr
)


g_r, _, _, S_k = lj_coul_binary.solve(rhos=[rho1, rho2])

r, g_r, U_r, k, S_k = lj_coul_binary.r, lj_coul_binary.g_r, lj_coul_binary.U_r, lj_coul_binary.k, lj_coul_binary.S_k
U_r_real = lj_coul_binary.U_r_erf_real
for i, j in np.ndindex(lj_coul_binary.n_components, lj_coul_binary.n_components):
    if not (i == j == 0):
        continue
    ax1.plot(r, np.abs(r * (g_r[i, j] - 1)),
             lw=1.5, ls=':', label='{}{}_ng'.format(i, j))
    ax2.plot(r, U_r[i, j], lw=1.5, ls='--', label='{}{}_ng'.format(i, j))
    ax2.plot(r, U_r_real[i, j], lw=1.5,ls='--',  label='{}{} erf'.format(i, j))
    ax2.plot(r, U_r[i, j] - U_r_real[i, j], ls='--', lw=1.5, label='{}{} short'.format(i, j))
    ax3.plot(k, S_k[i, j], lw=1.5, label='{}{}_ng'.format(i, j))
# import ipdb; ipdb.set_trace()

# Initialize a blank system and a Lennard-Jones potential with mixing rules.
lj_coul_binary = oz.System(kT=T, n_points=16384, dr=0.02)
r = lj_coul_binary.r
k = lj_coul_binary.k

coul_00 = oz.coulomb(r, q0, q0)
coul_01 = oz.coulomb(r, q0, q1)
coul_11 = oz.coulomb(r, q1, q1)

U_k_corr = exp(-1 * k**2 / (4 * 1.08**2)) / k**2
U_r_corr = erf(1.08 * r)

lj_coul_binary.set_interaction(
    0, 0,
    oz.wca(r, sig=sig0, eps=eps0, m=12, n=6) + coul_00,
    # oz.lennard_jones(r, sig=sig0, eps=eps0) + coul_00,
)
lj_coul_binary.set_interaction(
    0, 1,
    oz.wca(r, sig=sig01, eps=eps01, m=12, n=6) + coul_01,
    # oz.lennard_jones(r, sig=sig01, eps=eps01) + coul_01,
)
lj_coul_binary.set_interaction(
    1, 1,
    oz.wca(r, sig=sig1, eps=eps1, m=12, n=6) + coul_11,
    # oz.lennard_jones(r, sig=sig1, eps=eps1) + coul_11,
)


g_r, _, _, S_k = lj_coul_binary.solve(rhos=[rho1, rho2])

r, g_r, U_r, k, S_k = lj_coul_binary.r, lj_coul_binary.g_r, lj_coul_binary.U_r, lj_coul_binary.k, lj_coul_binary.S_k
U_r_real = lj_coul_binary.U_r_erf_real
for i, j in np.ndindex(lj_coul_binary.n_components, lj_coul_binary.n_components):
    if not (i == j == 0):
        continue
    ax1.plot(r, np.abs(r * (g_r[i, j] - 1)),
             lw=1.5, ls='--', label='{}{}'.format(i, j))
    ax1.plot(r[::100], -q0 * q1 * exp(-r[::100] / deb_l),
             ms=5, marker='o', lw=0, label='deb')
    ax2.plot(r, U_r[i, j], lw=1.5, ls=':', label='{}{}'.format(i, j))
    # ax2.plot(r, U_r_real[i, j], lw=1.5, label='{}{} erf'.format(i, j))
    # ax2.plot(r, U_r[i, j] - U_r_real[i, j], lw=1.5, label='{}{} short'.format(i, j))
    ax3.plot(k, S_k[i, j], lw=1.5, label='{}{}'.format(i, j))


lj_coul_binary = oz.System(kT=T, n_points=16384, dr=0.02)
r = lj_coul_binary.r
k = lj_coul_binary.k

coul_00 = oz.coulomb(r, q0, q0)
coul_01 = oz.coulomb(r, q0, q1)
coul_11 = oz.coulomb(r, q1, q1)

U_k_corr = exp(-1 * k**2 / (4 * 1.08**2)) / k**2
U_r_corr = erf(1.08 * r)

lj_coul_binary.set_interaction(
    0, 0,
    oz.wca(r, sig=sig0, eps=eps0, m=12, n=6) + coul_00,
    # oz.lennard_jones(r, sig=sig0, eps=eps0) + coul_00,
)
lj_coul_binary.set_interaction(
    0, 1,
    oz.wca(r, sig=sig01, eps=eps01, m=12, n=6) + coul_01,
    # oz.lennard_jones(r, sig=sig01, eps=eps01) + coul_01,
)
lj_coul_binary.set_interaction(
    1, 1,
    oz.wca(r, sig=sig1, eps=eps1, m=12, n=6) + coul_11,
    # oz.lennard_jones(r, sig=sig1, eps=eps1) + coul_11,
)


lj_coul_binary.U_r[0, 0] = lj_coul_binary.U_r[0, 0] - lj_coul_binary.U_r[0, 0, -1]
lj_coul_binary.U_r[0, 1] = lj_coul_binary.U_r[0, 1] - lj_coul_binary.U_r[0, 1, -1]
lj_coul_binary.U_r[1, 0] = lj_coul_binary.U_r[1, 0] - lj_coul_binary.U_r[1, 0, -1]
lj_coul_binary.U_r[1, 1] = lj_coul_binary.U_r[1, 1] - lj_coul_binary.U_r[1, 1, -1]

g_r, _, _, S_k = lj_coul_binary.solve(rhos=[rho1, rho2])

r, g_r, U_r, k, S_k = lj_coul_binary.r, lj_coul_binary.g_r, lj_coul_binary.U_r, lj_coul_binary.k, lj_coul_binary.S_k
U_r_real = lj_coul_binary.U_r_erf_real
for i, j in np.ndindex(lj_coul_binary.n_components, lj_coul_binary.n_components):
    if not (i == j == 0):
        continue
    ax1.plot(r, np.abs(r * (g_r[i, j] - 1)),
             lw=1.5, ls='--', label='{}{}_hack'.format(i, j))
    ax2.plot(r, U_r[i, j], lw=1.5, ls=':', label='{}{}_hack'.format(i, j))
    # ax2.plot(r, U_r_real[i, j], lw=1.5, label='{}{} erf'.format(i, j))
    # ax2.plot(r, U_r[i, j] - U_r_real[i, j], lw=1.5, label='{}{} short'.format(i, j))
    ax3.plot(k, S_k[i, j], lw=1.5, label='{}{}_hack'.format(i, j))


ax1.set_xlabel('r')
ax1.set_ylabel('h(r)')
ax1.legend(loc='upper right')

# ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_xlim((2, 100))
# ax1.set_ylim((1e-3, 1e-2))
# ax1.set_ylim((5 * 1e-1, 3))
fig1.savefig('figs/g_r_loglog.pdf', bbox_inches='tight')

ax1.set_xscale('linear')
ax1.set_yscale('linear')
# ax1.set_xlim((0, 15))
# ax1.set_ylim((-0.1, 3))
fig1.savefig('figs/g_r.pdf', bbox_inches='tight')

ax2.set_xlabel('r (Å)')
ax2.set_ylabel('U(r) (kT)')
ax2.legend(loc='upper right')
# ax2.set_ylim((-2.00, 1.00))
# ax2.set_yscale('log')
ax2.set_xlim((0, 30))
# ax2.set_ylim((-2, 2))
ax2.set_ylim((-.1, 0.5))
fig2.savefig('figs/U_r.pdf', bbox_inches='tight')

ax3.set_xlabel('k')
ax3.set_ylabel('S(k)')
ax3.legend(loc='lower right')
# ax3.set_xlim((0, 10))
ax3.set_xscale('log')
fig3.savefig('figs/S_k.pdf', bbox_inches='tight')
