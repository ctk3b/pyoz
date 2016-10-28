import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import numpy as np
import pyoz as oz
import matplotlib
from pyoz.exceptions import PyozError

import seaborn as sns

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

FS = 22
matplotlib.rc('xtick', labelsize=FS)
matplotlib.rc('ytick', labelsize=FS)


def dpd_func(r, a):
    cutoff = np.abs(r - 1.0).argmin()
    dpd = np.zeros_like(r)
    dpd[:cutoff] = 0.5 * a * (1 - r[:cutoff])**2
    return dpd


dpd_binary = oz.System()
r = dpd_binary.r
k = dpd_binary.k

a_ii = 15.0
chi = 2.2
chi_str = '$\chi =$ ' + str(chi)

a_ij = a_ii + chi/0.689
# print("a_ij = ", a_ij)


dpd_binary.set_interaction(0, 0, dpd_func(r, a=a_ii))
dpd_binary.set_interaction(0, 1, dpd_func(r, a=a_ij))
dpd_binary.set_interaction(1, 1, dpd_func(r, a=a_ii))

# dpd_binary.set_interaction(0, 0, oz.lennard_jones(r, 1, 1))
# dpd_binary.set_interaction(0, 1, oz.lennard_jones(r, 1, 1))
# dpd_binary.set_interaction(1, 1, oz.lennard_jones(r, 1, 1))

U_r = dpd_binary.U_r

plt.plot(r, U_r[0, 0], marker='D', markevery=10, label='0-0')
plt.plot(r, U_r[0, 1], marker='o', markevery=10, label='0-1')
plt.plot(r, U_r[1, 1], marker='x', markevery=10, label='1-1')
plt.xlabel('r', fontsize=FS)
plt.ylabel('U(r)', fontsize=FS)
plt.xlim(0, 1.5)
plt.legend()

#total density is fixed
rho_total = 5.0
# rho_total = 0.6

x0 = np.arange(0.005, 1.01, 0.005)
# x0 = np.arange(0.005,0.10,0.005)
x1 = 1.0-x0
nx = len(x0)
print(nx)

rho0 = rho_total * x0
rho1 = rho_total * x1

fig_gr = plt.figure(figsize=[18,10])
fig_gr00 = fig_gr.add_subplot(131)
fig_gr11 = fig_gr.add_subplot(132)
fig_gr01 = fig_gr.add_subplot(133)

fig_gr00.set_title('$g_{00}(r)$', fontsize=FS)
fig_gr11.set_title('$g_{11}(r)$; ' + chi_str, fontsize=FS)
fig_gr01.set_title('$g_{01}(r)$', fontsize=FS)

fig_gr00.set_xlim([0,5])
fig_gr11.set_xlim([0,5])
fig_gr01.set_xlim([0,5])


fig_Sk = plt.figure(figsize=[18,10])
fig_Sk_nn = fig_Sk.add_subplot(131)
fig_Sk_cc = fig_Sk.add_subplot(132)
fig_Sk_nc = fig_Sk.add_subplot(133)

fig_Sk_nn.set_xlim([0,25])
fig_Sk_nc.set_xlim([0,25])
fig_Sk_cc.set_xlim([0,25])

fig_Sk_nn.set_title('$S_{nn}(k)$', fontsize=FS)
fig_Sk_cc.set_title('$S_{cc}(k)$; ' + chi_str, fontsize=FS)
fig_Sk_nc.set_title('$S_{nc}(k)$', fontsize=FS)

e_r_guess = None

x_converged = []
S_K_nn_0 = []
S_K_nc_0 = []
S_K_cc_0 = []
P_virial = []

colors = sns.color_palette('viridis', len(range((nx))))
for i, color in zip(range(nx), colors):

    g_r, c_r, e_r, h_k = dpd_binary.solve(rhos=[rho0[i], rho1[i]], closure_name='hnc', initial_e_r=e_r_guess, max_iter=500)

    if np.isnan(g_r).all():
        e_r_guess = None
        continue
#        continue #goes to next iteration
    
    e_r_guess = e_r

    # print("type(gr) =",  type(g_r))
    
#    label = str(np.round(rho0[i],2))
    label = "$x_0$ = " + str(np.round(x0[i] * 100,2)) + "%"
    fig_gr00.plot(r,g_r[0,0],label=label, linewidth=2.0)
    fig_gr11.plot(r,g_r[1,1],label=label, linewidth=2.0)
    fig_gr01.plot(r,g_r[0,1],label=label, linewidth=2.0)

    S_K_nn = oz.structure_factors(dpd_binary, formalism='Bhatia-Thornton', combination='nn')
    S_K_cc = oz.structure_factors(dpd_binary, formalism='Bhatia-Thornton', combination='cc')
    S_K_nc = oz.structure_factors(dpd_binary, formalism='Bhatia-Thornton', combination='nc')

    x_converged.append(x0[i])
    S_K_nn_0.append(S_K_nn[0])
    S_K_nc_0.append(S_K_nc[0])
    S_K_cc_0.append(S_K_cc[0])

    # print(type(S_K_nn), np.shape(S_K_nn))

    fig_Sk_nn.plot(k,S_K_nn,label=label, linewidth=2.0, color=color)
    fig_Sk_cc.plot(k,S_K_cc,label=label, linewidth=2.0, color=color)
    fig_Sk_nc.plot(k,S_K_nc,label=label, linewidth=2.0, color=color)

    P_virial.append(oz.pressure_virial(dpd_binary))
    # print("P_virial = ", P_virial)

    

    # print("i = ", i)

fig_Sk_cc.set_xlabel('k', size=20)
fig_Sk_nc.set_xlabel('k', size=20)
fig_Sk_nn.set_xlabel('k', size=20)

np.save('sk_nn.npy', np.array(S_K_nn_0))
np.save('sk_cc.npy', np.array(S_K_cc_0))
np.save('sk_nc.npy', np.array(S_K_nc_0))



x_converged = np.array(x_converged)
#Plot long wavelength (k=0) limit of S(k)
fig_spinodal = plt.figure(figsize=[15,5])
fig_sp1 = fig_spinodal.add_subplot(131)
fig_sp2 = fig_spinodal.add_subplot(132)
fig_sp3 = fig_spinodal.add_subplot(133)

fig_sp2.set_title(chi_str, fontsize=FS)

fig_sp1.set_xlabel('$x_0$, %',fontsize=FS)
fig_sp2.set_xlabel('$x_0$, %',fontsize=FS)
fig_sp3.set_xlabel('$x_0$, %',fontsize=FS)

fig_sp1.set_ylabel('$S_{nn}(K=0)$',fontsize=FS)
fig_sp2.set_ylabel('$S_{cc}(K=0)$',fontsize=FS)
fig_sp3.set_ylabel('$S_{nc}(K=0)$',fontsize=FS)

fig_sp1.plot(x_converged * 100, S_K_nn_0, label='nn', marker='d', linewidth=2.0)
fig_sp2.plot(x_converged * 100, S_K_cc_0, label='cc', marker='d', linewidth=2.0)
fig_sp3.plot(x_converged * 100, S_K_nc_0, label='nc', marker='d', linewidth=2.0)


fig_spinodal.savefig('spinodal.pdf')
fig_gr00.legend(loc='lower right', ncol=1, fontsize=16)

fig_gr.savefig('gr.pdf')
fig_Sk_nn.legend(loc='lower right', ncol=1, fontsize=16)

fig_gr.tight_layout()
fig_Sk.tight_layout()
fig_spinodal.tight_layout()

fig_P = plt.figure()
fig_P1 = fig_P.add_subplot(111)
fig_P1.plot(x_converged, P_virial, marker='d', linewidth=2)
fig_P1.set_ylabel('$P_{virial}$', fontsize=FS)
fig_P1.set_xlabel('$x_0$, %', fontsize=FS)


fig_Sk.savefig('sk_lj.pdf')
# plt.show()

a = [15.0, 12.5, 10.0, 9.0, 8.5, 8.5, 8.5, 8.0, 7.8]
liq = [0.98673, 0.96946, 0.91453, 0.8556, 0.781, 0.7932, 0.7911, 0.6969, 0.652]
vap = [0.01333, 0.03066, 0.0853, 0.1456, 0.211, 0.2062, 0.2072, 0.317, 0.363]





