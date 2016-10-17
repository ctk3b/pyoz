from collections import OrderedDict
from copy import deepcopy
import glob
import pickle
import os
import shutil
import tempfile
from tempfile import mkdtemp

import numpy as np
import xarray as xr

import pyoz as oz
from pyoz.exceptions import PyozError


def init_thermo_array(variables, thermo_props):
    numbers_of_values = [len(values) for values in variables.values()]

    variables['thermo'] = thermo_props

    blank_data = np.empty(shape=(*numbers_of_values, len(thermo_props)))
    blank_data[:] = np.nan
    thermo_data = xr.DataArray(data=blank_data,
                               dims=tuple(variables.keys()),
                               coords=variables)
    return thermo_data


def init_raw_array(r, variables, raw_props):
    numbers_of_values = [len(values) for values in variables.values()]

    variables['data'] = raw_props
    variables['r'] = r

    blank_data = np.empty(shape=(*numbers_of_values, len(raw_props), len(r)))
    blank_data[:] = np.nan
    raw_data = xr.DataArray(data=blank_data,
                            dims=tuple(variables.keys()),
                            coords=variables)
    return raw_data


def run(x, rho, a_ii=1.0, a_jj=1.0, a_ij=1.0, kT=1, prefix='', output_dir=''):

    dr = 0.01
    syst = oz.System(kT=kT, dr=dr, n_points=8192)
    r = syst.r

    U_ii = oz.dpd(r, a_ii)
    U_jj = oz.dpd(r, a_jj)
    U_ij = oz.dpd(r, a_ij)

    syst.set_interaction(0, 0, U_ii)
    syst.set_interaction(1, 1, U_jj)
    syst.set_interaction(0, 1, U_ij)

    rhos = [x * rho, (1-x) * rho]

    # for mix in [0.8, 0.9, 0.7, 0.5]:
    for mix in [0.8]:
        try:
            g_r, c_r, e_r, H_k = syst.solve(
                rhos=rhos, closure_name='hnc', #reference_system=ref,
                mix_param=mix, status_updates=False, max_iter=5000)
        except PyozError as e:
            print(e)
            continue
        else:
            B2 = oz.second_virial_coefficient(syst)
            # print(eps, B2)
            # P_virial = oz.pressure_virial(syst)
            P_virial = np.nan
            # mu_ex = oz.excess_chemical_potential(syst)[0]
            # mu = mu_ex + kT * np.log(rho)
            mu = np.nan
            # s2 = oz.two_particle_excess_entropy(syst)[0]
            s2 = np.nan
            Snn = oz.structure_factors(syst, formalism='bt', combination='nn')
            Snc = oz.structure_factors(syst, formalism='bt', combination='nc')
            Scc = oz.structure_factors(syst, formalism='bt', combination='cc')
            S_k = oz.structure_factors(syst, formalism='fz')
            break
    else:
        g_r = c_r = e_r = S_k = np.empty_like(syst.U_r)
        g_r[:] = c_r[:] = e_r[:] = S_k[:] = np.nan

        Snn = Snc = Scc = np.empty_like(syst.r)
        Snn[:] = Snc[:] = Scc[:] = np.nan

        B2 = P_virial = mu = s2 = np.nan

    # U_r = syst.U_r[0, 0]
    data = {'r': syst.r,
            'k': syst.k,
            'g_r_00': g_r[0, 0],
            'g_r_01': g_r[0, 1],
            'g_r_11': g_r[1, 1],
            'c_r': c_r,
            'e_r': e_r,
            # 'U_r': U_r,
            'Snn': Snn,
            'Snc': Snc,
            'Scc': Scc,
            'S_k_00': S_k[0, 0],
            'S_k_01': S_k[0, 1],
            'S_k_11': S_k[1, 1],
            'B2': B2,
            'P_virial': P_virial,
            'mu': mu,
            's2': s2,
            'rho': rho,
            'kT': kT,
            'x': x}
    file_name = '{}x_{:.5f}-rho_{:.3f}.pkl'.format(prefix, x, rho)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # === VARIABLES === #
    d_epsilon = 0.05
    a_ii = 15.0
    a_jj = 31.6
    a_ij = 27.2

    rho = 3.0

    xs = np.arange(0, 1, 0.01)

    temps = np.arange(1.0, 1.5, 0.01)

    variables = OrderedDict([('kT', temps),
                             ('x', xs)])

    print('n_runs', np.product([len(x) for x in variables.values()]))
    # === VARIABLES === #
    # with tempfile.TemporaryDirectory(prefix='pyoz_') as pkl_dir:
    pkl_dir = 'scan'
    eps = 1.0
    eps_cross = 0.95
    run_prefix = 'two_comp_DPD-{:.2f}-{:.2f}-{:.2f}'.format(a_ii, a_jj, a_ij)

    from distributed import Client
    client = Client()
    data = [client.submit(run,
                          x=x,
                          rho=3.0,
                          a_ii=a_ii,
                          a_jj=a_jj,
                          a_ij=a_ij,
                          kT=kT,
                          prefix=run_prefix,
                          output_dir=pkl_dir)
            for x in xs
            for kT in temps]

    # data = progress(data)
    data = client.gather(data)

    pkls = os.path.join(pkl_dir, '*.pkl'.format(run_prefix))
    all_files = glob.glob(pkls)

    with open(all_files[0], 'rb') as f:
        data = pickle.load(f)
        r = data['r']

    properties = ['B2', 'P_virial', 'mu', 's2']
    thermo_data = init_thermo_array(deepcopy(variables), properties)
    distributions = ['g_r_00', 'g_r_01', 'g_r_11',
                     'S_k_00', 'S_k_01', 'S_k_11',
                     'Snn',
                     'Snc',
                     'Scc']
    raw_data = init_raw_array(r, deepcopy(variables), distributions)
    for n, file_name in enumerate(all_files):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        idx = tuple(data[x] for x in variables.keys())
        props = [data[x] for x in properties]
        thermo_data.loc[idx] = props

        distr = np.array([data[x] for x in distributions])
        raw_data.loc[idx] = distr

    # Store dat stuff
    os.makedirs('data', exist_ok=True)
    ds = thermo_data.to_dataset(name='THERMO')
    path = 'data/{}thermo_data.nc'.format(run_prefix)
    ds.to_netcdf(path)

    ds = raw_data.to_dataset(name='RAW')
    path = 'data/{}thermo_data.nc'.format(run_prefix)
    ds.to_netcdf(path)

    # shutil.rmtree(pkl_dir)
    print('FINITO')

