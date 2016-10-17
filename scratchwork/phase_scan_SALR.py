from collections import OrderedDict
from copy import deepcopy
import glob
import pickle
import os

from distributed import Client
from distributed.diagnostics import progress
import numpy as np
import xarray as xr

import pyoz as oz
import pyoz.unit as u
from pyoz.exceptions import PyozError


def number_densities(SiO2_wt_perc):
    # SiO2_mol_wt = 60.08 * u.gram / u.mole
    # rho_solution = 1 * u.gram / u.centimeter**3
    # colloid_rho = rho_solution * (SiO2_wt_perc/100) / SiO2_mol_wt * Na
    solution_density = 1

    # sol_dens = () / (() + ())
    return 6 / np.pi * (SiO2_wt_perc / 100) / (2.65 / solution_density)


def LR_parameters(colloid_diameter, NaCl_wt_perc, Z):
    NaCl_mol_wt = 58.44 * u.gram / u.mole
    rho_solution = 1 * u.gram / u.centimeter**3

    # NaCl_conc = NaCl_wt_perc * 10.0 / NaCl_mol_wt  # mol/Liter
    # NaCl_conc = rho_solution * (NaCl_wt_perc/100) / NaCl_mol_wt
    NaCl_conc = (NaCl_wt_perc/100) *10 / 58.4
    # ionic_strength = NaCl_conc.value_in_unit(u.moles / u.liter)

    # lambda_d = 0.304 / np.sqrt(ionic_strength) * u.nanometers
    lambda_d = 0.304 / np.sqrt(NaCl_conc) * u.nanometers
    lambda_b = 0.70 * u.nanometers


    l_debye = lambda_d / colloid_diameter  # Reduced units
    l_bjerrum = lambda_b / colloid_diameter  # Reduced units


    # Charge density of glass is given for a few different pH values:
    # http://physics.nyu.edu/grierlab/charge6c/
    q_density = Z * u.elementary_charge / u.micrometers**2
    colloid_area = np.pi * colloid_diameter**2
    colloid_q = q_density * colloid_area

    #Equation 4: Bollinger, 2016 paper.
    # A/kT actually
    A = colloid_q**2 * l_bjerrum / (1 + 0.50 / l_debye)**2

    # TODO: Is there another unit reduction necessary here?
    A = A.value_in_unit(u.elementary_charge**2)
    return A, l_debye


def SALR(r, eps, m, n, A, l_debye):
    SR = 4 * r**(-m)
    SA = -4 * eps * r**(-n)
    LR = A * np.exp(-(r - 1) / l_debye)
    return SR + SA + LR


def SA_ref(r, eps, m, n):
    p = 1 / (m - n)
    r_cut = (m / eps / n)**p
    # r_cut = np.exp((np.log(m)-np.log(n)-np.log(eps)) / (m - n))
    U = SALR(r, eps, m, n, A=0, l_debye=1)
    # depth = SALR(r_cut, eps, m, n, A=0, l_debye=1)
    U = U - np.nanmin(U)
    return np.where(r < r_cut, U, 0)


def init_thermo_array(variables):
    numbers_of_values = [len(values) for values in variables.values()]

    thermo_props = ['B2', 'P_virial', 'mu', 'Sk0', 's2']
    variables['thermo'] = thermo_props

    blank_data = np.empty(shape=(*numbers_of_values, len(thermo_props)))
    blank_data[:] = np.nan
    thermo_data = xr.DataArray(data=blank_data,
                               dims=tuple(variables.keys()),
                               coords=variables)
    return thermo_data


def init_raw_array(r, variables):
    numbers_of_values = [len(values) for values in variables.values()]
    # r = data[0]['r']
    # raw_props = ['g_r', 'g_r_ref', 'S_k', 'S_k_ref']
    raw_props = ['g_r', 'S_k', 'U_r']
    variables['data'] = raw_props
    variables['r'] = r

    blank_data = np.empty(shape=(*numbers_of_values, len(raw_props), len(r)))
    blank_data[:] = np.nan
    raw_data = xr.DataArray(data=blank_data,
                            dims=tuple(variables.keys()),
                            coords=variables)

    return raw_data


def run(eps, NaCl_wt_perc, SiO2_wt_perc, Z, m=100, n=50, T=298, prefix=''):
    # NOTE: Temperatures other than 298 K invalidate the Debye length
    # approximation used: kappa^-1 = 0.304 / sqrt(I(M))
    dr = 0.01
    syst = oz.System(kT=1, dr=dr, n_points=8192)
    d = 40 * u.nanometer
    A, l_debye = LR_parameters(colloid_diameter=d,  NaCl_wt_perc=NaCl_wt_perc,
                               Z=Z)
    # A = 0
    salr = SALR(r=syst.r, eps=eps, m=m, n=n,
               A=A,
               l_debye=l_debye)
    # salr_ref = SA_ref(r=syst.r, eps=1, m=m, n=n)
    # wca = oz.wca(syst.r, eps=1, sig=1, m=m, n=n)
    rho = number_densities(SiO2_wt_perc=SiO2_wt_perc)

    # for mix in [0.8, 0.9, 0.7, 0.5]:
    for mix in [0.8]:
        # syst = oz.System(kT=1 / eps, dr=dr, n_pts=8192)
        # syst.set_interaction(0, 0, salr_ref)
        # try:
        #     g_r, c_r, e_r, S_k = syst.solve(
        #         rhos=rho, closure_name='hnc', mix_param=0.8,
        #         status_updates=False)
        # except PyozError as e:
        #     print('WCA failed', e)
        #     continue
        # U_r_ref = syst.U_r
        # g_r_ref = g_r
        # S_k_ref = S_k
        #
        # syst = oz.System(kT=1 / eps, dr=dr, n_pts=8192)
        # syst.set_interaction(0, 0, salr)
        # try:
        #     g_r, c_r, e_r, S_k = syst.solve(
        #         rhos=rho, closure_name='rhnc', mix_param=mix,
        #         status_updates=False, max_iter=5000,
        #         initial_e_r=e_r, e_r_ref=e_r, U_r_ref=U_r_ref, g_r_ref=g_r_ref)
        # except PyozError as e:
        #     print('Mix', mix, e)
        #     continue
        syst = oz.System(kT=1, dr=dr, n_points=8192)
        syst.set_interaction(0, 0, salr)
        try:
            g_r, c_r, e_r, S_k = syst.solve(
                rhos=rho, closure_name='hnc', mix_param=0.8,
                status_updates=False)
        except PyozError as e:
            print('LJ failed', e)
            continue
        else:
            B2 = oz.second_virial_coefficient(syst)
            # print(eps, B2)
            P_virial = oz.pressure_virial(syst)
            mu_ex = oz.excess_chemical_potential(syst)[0]
            mu = mu_ex + T * np.log(rho)
            s2 = oz.two_particle_excess_entropy(syst)[0]
            break
    else:
        # g_r = c_r = e_r = S_k = B2 = P_virial = mu = s2 = g_r_ref = S_k_ref = np.nan
        g_r = c_r = e_r = S_k = B2 = P_virial = mu = s2 = np.nan

    if g_r is np.nan:
        g_r = np.empty_like(syst.r)
        g_r[:] = np.nan
    else:
        g_r = g_r[0, 0]

    # if g_r_ref is np.nan:
    #     g_r_ref = np.empty_like(syst.r)
    #     g_r_ref[:] = np.nan
    # else:
    #     g_r_ref = g_r_ref[0, 0]

    if c_r is np.nan:
        c_r = np.empty_like(syst.r)
        c_r[:] = np.nan
    else:
        c_r = c_r[0, 0]

    if e_r is np.nan:
        e_r = np.empty_like(syst.r)
        e_r[:] = np.nan
    else:
        e_r = e_r[0, 0]

    if S_k is np.nan:
        S_k = np.empty_like(syst.r)
        S_k[:] = np.nan
    else:
        S_k = S_k[0, 0]

    # if S_k_ref is np.nan:
    #     S_k_ref = np.empty_like(syst.r)
    #     S_k_ref[:] = np.nan
    # else:
    #     S_k_ref = S_k_ref[0, 0]

    U_r = syst.U_r[0, 0]
    data = {'r': syst.r,
            'k': syst.k,
            'g_r': g_r,
            # 'g_r_ref': g_r_ref,
            'c_r': c_r,
            'e_r': e_r,
            'U_r': U_r,
            'S_k': S_k,
            # 'S_k_ref': S_k_ref,
            'B2': B2,
            'P_virial': P_virial,
            'mu': mu,
            's2': s2,
            'Sk0': S_k[0],
            'rho': rho,
            'SiO2': SiO2_wt_perc,
            'NaCl': NaCl_wt_perc,
            'epsilon': eps,
            'A': A,
            'Z': Z,
            'm': m,
            'n': n,
            'l_debye': l_debye}
    # fn = 'scan/eps_{:.5f}-sio2_{:.3f}-nacl_{:.3f}-n_{}.pkl'.format(eps, SiO2_wt_perc, NaCl_wt_perc, n)
    fn = 'scan/{}eps_{:.5f}-Z_{:.0e}-rho_{:.3f}-m_{}-n_{}.pkl'.format(prefix, eps, Z, rho, m, n)
    with open(fn, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    client = Client()

    # === VARIABLES === #
    d_epsilon = 0.01
    # epsilons = np.arange(1.6, 0.7, d_epsilon)
    epsilons = np.arange(0.5, 5.0, d_epsilon)
    epsilons = [1.829]  # 2
    # epsilons = [2.36345, 2.41345, 2.46345, 2.51345, 2.52345, 2.53345, 2.54345, 2.55345, 2.56345]  # 1
    # epsilons = [2.46345]  # 1
    # epsilons = [5, 6] # 0.5
    # epsilons = [4]

    Zs = [1e3, 5e3, 1e4, 5e4, 1e5]

    d_NaCl_wt_perc = 0.1
    NaCl_wt_perc = np.arange(0.10, 10, d_NaCl_wt_perc)
    # NaCl_wt_perc = [5e-1, 1, 2, 5, 10]
    NaCl_wt_perc = [2.0]
    # NaCl_wt_perc = [0.25]

    d_SiO2_wt_perc = 1
    # SiO2_wt_perc = np.arange(1, 20, d_SiO2_wt_perc)
    SiO2_wt_perc = np.arange(1, 100, d_SiO2_wt_perc)
    # SiO2_wt_perc = [1]

    # ms = np.arange(12, 97, 12)
    # ns = np.arange(5, 36, 5)

    # mn = list(zip([50, 50, 50, 100, 100], [6, 12, 18, 12, 24]))
    # mn = [(100, 50)]
    mn = [(50, 18)]

    variables = OrderedDict([('SiO2', SiO2_wt_perc),
                             ('NaCl', NaCl_wt_perc),
                             ('epsilon', epsilons),
                             ('Z', Zs),
                             ('mn', ['{}-{}'.format(m, n) for m, n in mn])])

    print('n_runs', np.product([len(x) for x in variables.values()]))
    # === VARIABLES === #
    run_prefix = 'boyle_Z_'
    data = [client.submit(run,
                          eps=eps,
                          # NaCl_wt_perc=1,
                          NaCl_wt_perc=nacl,
                          SiO2_wt_perc=sio2,
                          Z=Z,
                          m=m,
                          n=n,
                          prefix=run_prefix)
            for eps in epsilons
            for Z in Zs
            for sio2 in SiO2_wt_perc
            for nacl in NaCl_wt_perc
            for m, n in mn]

    # data = progress(data)
    data = client.gather(data)

    os.makedirs('scan', exist_ok=True)
    all_files = glob.glob('scan/{}eps*Z*.pkl'.format(run_prefix))

    with open(all_files[0], 'rb') as f:
        data = pickle.load(f)
        r = data['r']

    thermo_data = init_thermo_array(deepcopy(variables))
    raw_data = init_raw_array(r, deepcopy(variables))
    for n, file_name in enumerate(all_files):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        # idx = tuple(data[x] for x in ('SiO2 wt%', 'NaCl wt%', 'epsilon', 'n'))
        idx = tuple([data['SiO2'],
                     data['NaCl'],
                     data['epsilon'],
                     data['Z'],
                     '{}-{}'.format(data['m'], data['n'])])
        props = [data[x] for x in ['B2', 'P_virial', 'mu', 'Sk0', 's2']]
        thermo_data.loc[idx] = props

        # distr = [data[x] for x in ['g_r', 'g_r_ref', 'S_k', 'S_k_ref']]
        distr = [data[x] for x in ['g_r', 'S_k', 'U_r']]
        raw_data.loc[idx] = distr

    # Store dat stuff
    os.makedirs('data', exist_ok=True)
    ds = thermo_data.to_dataset(name='THERMO')
    ds.to_netcdf('data/{}thermo_data.nc'.format(run_prefix))

    ds = raw_data.to_dataset(name='RAW')
    ds.to_netcdf('data/{}raw_data.nc'.format(run_prefix))

    for file_name in all_files:
        os.remove(file_name)

