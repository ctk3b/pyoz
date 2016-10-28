import numpy as np

import pyoz as oz

alpha = 100
epsilon = 100
A = 0.1
xi = 2

d_rho = 0.1
rhos = np.arange(0, 1, d_rho)
T = 1.0


def SALR(r, a, e, A, xi):
    return 4 * e * (r**(-2*a) - r**(-a)) + A * np.exp(-r/xi) / (r/xi)

# for rho_ij in rhos:
rho = 0.001
syst = oz.System(kT=T)
syst.set_interaction(0, 0, SALR(syst.r, 6, 1, 0.5, 40))
syst.solve(rho, closure_name='HNC')
