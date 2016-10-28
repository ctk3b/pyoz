import pyoz as oz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import leastsq

FS = 15
matplotlib.rc('xtick', labelsize=FS)
matplotlib.rc('ytick', labelsize=FS)

colormap_I = plt.get_cmap('Purples')
colormap_II = plt.get_cmap('Oranges')


def DPDPotential(a,r):
    U_DPD = np.where(r < 1, 0.5 * a  * (1-r)**2 , 0)
    return U_DPD

def CalcU_Perturbation(a,r, eps, form='I'):
    a_att = a/8.0 + eps
    U_p = a_att * np.sin(np.pi * r)**2
    U_perturbation = np.where(r<1.0, U_p, 0)
    if form == 'I':
        return U_perturbation
    elif form == 'II':
        U0 = np.where(r < 0.5, a_att, 0)
        U1 = np.where(r > 0.5, U_perturbation, 0)
        U_perturbation_WCA  = U0 + U1
        return U_perturbation_WCA

    else:
        print("Unknown form of  potential. Has to be I or II")
        return np.nan

def CalcU_DPD_Attractive(a,r, eps, form='I'):
    U_total = DPDPotential(a,r) - CalcU_Perturbation(a,r,eps, form)
    return U_total

#-------------------

def Calc_P_U(rho, a, eps, P_target, U_target, e_r_guess=None):
    dpd_att = oz.System()
    r = dpd_att.r
    U_att = CalcU_DPD_Attractive(a,r, eps, form='II')
    dpd_att.set_interaction(0,0,U_att)
    #calculate B2
    B2 = oz.properties.second_virial_coefficient(dpd_att)
    #Solve g_r
    g_r, c_r, e_r, S_k = dpd_att.solve(rhos=rho, closure_name='kh',
                                       initial_e_r = e_r_guess)
    P_virial = oz.properties.pressure_virial(dpd_att)

    U = oz.properties.internal_energy(dpd_att)
    #U = 0.0
    return B2, P_virial, U, r, g_r, e_r

def Error(params, P_target, U_target, w_p=0.5):
    a, eps = params
    P_virial, U, r, g_r[0,0,:] = Calc_P_U(rho, a, eps, P_target, U_target)
    Error = w_p * np.abs(1.0 - P_virial/P_target) + (1.0-w_p) * np.abs(1.0 - U/U_target)
    return Error





#------------------------------------
Na = 6.023e23
R = 8.314
T = 300.0
k = R/Na
P_atm = 1e5 #Pascals

RT = R*T
kT = k*T

U_cohesion = -42e3  #J/mol
rho = 5.0 #target density
r_c = 7.66e-10 #meters

P_target = P_atm * (r_c**3)/ kT
U_target = -U_cohesion
#-----------------------
rho = 5.0
a = 16.0

#First, simply solve this attractive system for different epsilon.
dpd_att = oz.System()
r = dpd_att.r

eps_array = np.arange(-a/8.0, 1.0, 0.25)
eps_array = np.array([-0.1])

fig_ur = plt.figure()
fig_ur1 = fig_ur.add_subplot(111)
fig_ur1.set_xlabel('$r$', fontsize=FS)
fig_ur1.set_ylabel('$U(r)$', fontsize=FS)
fig_ur1.set_xlim([0,1.2])

fig_gr = plt.figure()
fig_gr1 = fig_gr.add_subplot(111)
fig_gr1.set_xlabel('$r$', fontsize=FS)
fig_gr1.set_ylabel('$g(r)$', fontsize=FS)
fig_gr1.set_xlim([0,4])

#-----
fig_pu = plt.figure(figsize=[16,5])

fig_b2 = fig_pu.add_subplot(131)
fig_b2.set_xlabel('$\epsilon$', fontsize=FS)
fig_b2.set_ylabel('$B_2$', fontsize=FS)

fig_p = fig_pu.add_subplot(132)
fig_p.set_xlabel('$\epsilon$', fontsize=FS)
fig_p.set_ylabel('$P$', fontsize=FS)

fig_u = fig_pu.add_subplot(133)
fig_u.set_xlabel('$\epsilon$', fontsize=FS)
fig_u.set_ylabel('$U$', fontsize=FS)

P_array = []
U_array = []
B2_array = []
eps_converged_array = []

e_r_guess = None
for eps in eps_array:
    U_att = CalcU_DPD_Attractive(a,r, eps, form='II')
    fig_ur1.plot(r,U_att,linewidth=2.0, label=str(eps))
    try:
        B2, P, U, r, g_r, e_r = Calc_P_U(rho, a, eps, P_target, U_target, e_r_guess)
    except:
        continue

    e_r_guess = e_r
    fig_gr1.plot(r,g_r[0,0,:] , label = str(eps), linewidth=2.0)
    P_array.append(P)
    U_array.append(U)
    B2_array.append(B2)
    eps_converged_array.append(eps)

print(len(eps_converged_array), len(P_array), len(U_array))
print(eps_array)
print(P_array)
print(U_array)

fig_b2.plot(eps_converged_array, B2_array, marker='d', linewidth = 2.0)
fig_b2.axhline(0,linestyle='--', color='black', linewidth=2)

fig_p.plot(eps_converged_array, P_array, marker='d', linewidth = 2.0)
fig_u.plot(eps_converged_array, U_array, marker='d', linewidth = 2.0)

print("P_array = ", P_array)
fig_gr1.legend(loc='lower right')
fig_ur1.legend(loc='upper right')

fig_pu.tight_layout()
plt.show()