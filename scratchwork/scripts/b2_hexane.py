import pyoz as oz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import newton
import sys

FS = 15
matplotlib.rc('xtick', labelsize=FS)
matplotlib.rc('ytick', labelsize=FS)

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

def Solve_DPD_Att(rho, a, eps, e_r_guess = None):
    dpd_att = oz.System()
    r = dpd_att.r
    U_att = CalcU_DPD_Attractive(a,r, eps, form='II')
    dpd_att.set_interaction(0,0,U_att)
    g_r, c_r, e_r, H_k = dpd_att.solve(rhos=rho, closure_name='hnc', initial_e_r = e_r_guess)
    return dpd_att

def Calc_P_U(dpd_att):
    try:
        P_virial = oz.properties.pressure_virial(dpd_att)
    except TypeError:
        P_virial = -1

    try:
        U = oz.properties.internal_energy(dpd_att)
    except TypeError:
        U = -1
    return P_virial, U

def Error(params, P_target, U_target, w_p=0.5):
    a, eps = params
    try:
        dpd_att = Solve_DPD_Att(rho, a, eps)
    except:
        return -1

    P_virial, U = Calc_P_U(dpd_att)
    Error = w_p * np.abs(1.0 - P_virial/P_target) + (1.0-w_p) * np.abs(1.0 - U/U_target)
    return Error

def CalcB2(a,eps, form):
    dpd_att = oz.System()
    r = dpd_att.r
    U_att = CalcU_DPD_Attractive(a,r, eps,form)
    dpd_att.set_interaction(0,0,U_att)
    B2 = oz.properties.second_virial_coefficient(dpd_att)
    return B2

def B2Error(params, B2_target, form):
    a, eps = params
    B2 = CalcB2(a,eps,form)
    print("Func = B2Error; B2 = ", B2)
    B2Error = (1.0 - B2/B2_target)**2
    return B2Error

def CalcB2_nondim(B2_Dortmund, MW, density):
    '''
    density: g/cc
    MW: g/mol
    B2 = cc/mol
    '''
    B2_nondim = B2_Dortmund * density/MW
    return B2_nondim

#------------------------------------

dpd_att = oz.System()
r = dpd_att.r
k = dpd_att.k

#Hexane:
MW = 86
Density = 0.655 #g/cc
B2_Dortmund = -1729 #cm^3/mol
B2_nondim = CalcB2_nondim(B2_Dortmund, MW, Density)
print("B2_nondim = ", B2_nondim)

a = 30.0
eps = 2.5
params_init_guess = (a,eps)
print("B2Error = " + str(np.sqrt(B2Error(params_init_guess, B2_nondim, 'I'))))
#popt, pcov = newton(B2Error, params_init_guess, args=(B2_nondim))
#print(popt)


#sys.exit()

#Plot the potential
U_att = CalcU_DPD_Attractive(a,r, eps, form='I')
dpd_att.set_interaction(0,0,U_att)
B2 = oz.properties.second_virial_coefficient(dpd_att)
B2 = np.round(B2,2)

U_min = np.round(np.min(U_att),2)
fig_ur = plt.figure()
fig_ur1 = fig_ur.add_subplot(111)
fig_ur1.plot(r, U_att, linewidth=2.0)
fig_ur1.set_xlim([0,1.1])
fig_ur1.set_title('eps = ' + str(eps) + '; Umin = ' + str(U_min) + "; B2 = " + str(B2))

#----
rho_array = np.arange(25, 5, -0.50)

fig = plt.figure(figsize=[12,5])
fig_P = fig.add_subplot(121)
fig_U = fig.add_subplot(122)

fig_P.set_xlabel('$\\rho$', fontsize=FS)
fig_P.set_ylabel('$P$', fontsize=FS)

fig_U.set_xlabel('$\\rho$', fontsize=FS)
fig_U.set_ylabel('$U$', fontsize=FS)


rho_converged  = []
P_array = []
U_array = []

for rho in rho_array:

    print("rho = ", rho)
    U_att = CalcU_DPD_Attractive(a,r, eps, form='I')
    dpd_att = Solve_DPD_Att(rho, a, eps) #returns system object

    if dpd_att.g_r is None: #not converged
        pass
    else:
        #calculate B2
        B2 = oz.properties.second_virial_coefficient(dpd_att)
        #Calculate pressure and internal energy
        P, U = Calc_P_U(dpd_att) #calculate pressure and internal energy

        rho_converged.append(rho)
        P_array.append(P)
        U_array.append(U)

fig_P.plot(rho_converged, P_array, marker='d', color='blue')
fig_U.plot(rho_converged, U_array, marker='d', color='blue')
fig.tight_layout()

plt.show()








