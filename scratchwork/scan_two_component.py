from collections import OrderedDict
from copy import deepcopy
import glob
import pickle
import os

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


def run(x, rho, m=50, n=18, kT=1, prefix=''):
    dr = 0.01
    syst = oz.System(kT=kT, dr=dr, n_points=8192)
    r = syst.r
    like = oz.mie(r, eps=1, sig=1, m=m, n=n)
    cross = oz.mie(r, eps=0.5, sig=1, m=m, n=n)
    rhos = [x * rho, (1-x) * rho]

    # for mix in [0.8, 0.9, 0.7, 0.5]:
    for mix in [0.8]:
        syst.set_interaction(0, 0, like)
        syst.set_interaction(1, 1, like)
        syst.set_interaction(0, 1, cross)
        try:
            g_r, c_r, e_r, H_k = syst.solve(
                rhos=rhos, closure_name='hnc', mix_param=mix,
                status_updates=True, max_iter=5000)
        except PyozError as e:
            print('Mix', mix, e)
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
            break
    else:
        g_r = c_r = e_r = Snn = Snc = Scc = np.empty_like(syst.r)
        g_r[:] = c_r[:] = e_r[:] = Snn[:] = Snc[:] = Scc[:] = np.nan
        B2 = P_virial = mu = s2 = np.nan

    U_r = syst.U_r[0, 0]
    data = {'r': syst.r,
            'k': syst.k,
            'g_r': g_r,
            'c_r': c_r,
            'e_r': e_r,
            'U_r': U_r,
            'Snn': Snn,
            'Snc': Snc,
            'Scc': Scc,
            'B2': B2,
            'P_virial': P_virial,
            'mu': mu,
            's2': s2,
            'rho': rho,
            'x': x}
    # fn = 'scan/eps_{:.5f}-sio2_{:.3f}-nacl_{:.3f}-n_{}.pkl'.format(eps, SiO2_wt_perc, NaCl_wt_perc, n)
    # fn = 'scan/{}eps_{:.5f}-Z_{:.0e}-rho_{:.3f}-m_{}-n_{}.pkl'.format(prefix, eps, Z, rho, m, n)
    fn = 'scan/{}x_{:.5f}-rho_{:.3f}.pkl'.format(prefix, x, rho)
    with open(fn, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # === VARIABLES === #
    d_epsilon = 0.05
    epsilons = np.arange(0.5, 5.0, d_epsilon)
    epsilons = [1]

    d_rho = 0.05
    rhos = np.arange(0, 1.0)
    rhos = [0.65]

    xs = [0.5]

    variables = OrderedDict([('rho', rhos),
                             ('x', xs)])

    print('n_runs', np.product([len(x) for x in variables.values()]))

    # === VARIABLES === #
    run_prefix = 'two_comp_'

    from distributed import Client
    client = Client()
    data = [client.submit(run,
                          x=x,
                          rho=rho,
                          m=50,
                          n=18,
                          prefix=run_prefix)
            for x in xs
            for rho in rhos]

    # data = progress(data)
    data = client.gather(data)

    os.makedirs('scan', exist_ok=True)
    all_files = glob.glob('scan/*.pkl'.format(run_prefix))

    with open(all_files[0], 'rb') as f:
        data = pickle.load(f)
        r = data['r']

    properties = ['B2', 'P_virial', 'mu', 's2']
    thermo_data = init_thermo_array(deepcopy(variables), properties)
    distributions = ['g_r', 'Snn', 'Snc', 'Scc', 'U_r']
    raw_data = init_raw_array(r, deepcopy(variables), distributions)
    for n, file_name in enumerate(all_files):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        idx = tuple(data[x] for x in variables.keys())
        props = [data[x] for x in properties]
        thermo_data.loc[idx] = props

        distr = [data[x] for x in distributions]
        raw_data.loc[idx] = distr

    # Store dat stuff
    os.makedirs('data', exist_ok=True)
    ds = thermo_data.to_dataset(name='THERMO')
    ds.to_netcdf('data/{}thermo_data.nc'.format(run_prefix))

    ds = raw_data.to_dataset(name='RAW')
    ds.to_netcdf('data/{}raw_data.nc'.format(run_prefix))

    # for file_name in all_files:
    #     os.remove(file_name)

