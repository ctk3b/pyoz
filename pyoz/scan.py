from collections import OrderedDict
import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import pyoz as oz
from pyoz.exceptions import PyozWarning, PyozError

plt.style.use('fivethirtyeight')


def scan(variables, interactions, system_vars=None, plot=True):
    # Messes up indexing later
    assert isinstance(variables, OrderedDict)
    assert len(interactions[0]) == 3

    # System.solve() requires this. Technically could use default value but
    # there really aren't any non-testing scenarios where you do not want to
    # set this.
    assert 'rho' in variables

    if system_vars is None:
        system_vars = dict()
    syst = oz.System(**system_vars)
    thermo_data, raw_data = generate_data_storage(syst.r, variables)

    # TODO: better check

    if plot:
        fig_gr, ax_gr = plt.subplots()
        fig_ur, ax_ur = plt.subplots()
        fig_sk, ax_sk = plt.subplots()

    G_r = None
    vars = list(variables)
    states = [dict(zip(vars, p)) for p in it.product(*variables.values())]
    for state in states:
        label = format_state_point(state)
        oz.logger.info(label)

        syst = oz.System(**system_vars)
        for idx1, idx2, potential_func in interactions:
            U_r = potential_func(r=syst.r, **state)
            syst.set_interaction(idx1, idx2, U_r)


        rho = state['rho']
        T = syst.T
        try:
            g_r, c_r, G_r, S_k = syst.solve(
                rhos=rho, closure_name='hnc', mix_param=1.0,
                status_updates=True, initial_G_r=G_r, max_iter=5000)
        except PyozError as e:
            oz.logger.info(e)
            continue

        # Plot reference state.
        r, k, U_r = syst.r, syst.k, syst.U_r
        if plot:
            ax_gr.plot(r, g_r[0, 0], label=label)
            ax_ur.plot(r, U_r[0, 0], label=label)
            ax_sk.plot(k, S_k[0, 0], label=label)

        raw = {'g_r': g_r, 'c_r': c_r, 'G_r': G_r, 'U_r': U_r, 'S_k': S_k}
        for name, values in raw.items():
            sel = (*state.values(), name)
            raw_data.loc[sel] = values[0, 0]

        B2 = oz.second_virial_coefficient(syst)
        P_virial = oz.pressure_virial(syst)
        mu_ex = T * oz.excess_chemical_potential(syst)[0]
        mu = mu_ex + T * np.log(rho)
        s2 = oz.two_particle_excess_entropy(syst)

        sel = tuple(*state.values())
        thermo_data.loc[sel] = [B2, P_virial, mu, S_k[0, 0, 0], s2]

    format_and_save_figures(fig_gr, ax_gr,
                            fig_ur, ax_ur,
                            fig_sk, ax_sk,
                            xlim=(0, 5),
                            ur_ylim=(np.min(U_r) - 1, np.mean(U_r) + 3))
    return thermo_data, raw_data


def format_state_point(state):
    fmt = list()
    for var, value in state.items():
        try:
            int(value)
        except TypeError:
            for val in value:
                fmt.append('{}={:.3e}'.format(var, val))
        else:
            fmt.append('{}={:.3e}'.format(var, value))
    return ', '.join(fmt)


def generate_data_storage(r, variables):
    numbers_of_values = [len(values) for values in variables.values()]

    # TODO: Proper warning and change user defined name
    assert 'thermo' not in variables

    thermo_props = ['B2', 'P_virial', 'mu', 'Sk0', 's2']
    variables['thermo'] = thermo_props
    data = np.empty(shape=(*numbers_of_values, len(thermo_props)))
    # HACK FOR THE TIME BEING
    A = [x for x, _ in variables['LR_parms']]

    from copy import deepcopy
    temp_vars = deepcopy(variables)
    temp_vars['LR_parms'] = A
    data[:] = np.nan
    thermo_data = xr.DataArray(data=data,
                               dims=tuple(temp_vars.keys()),
                               coords=temp_vars)
    del variables['thermo']

    raw_props = ['g_r', 'c_r', 'G_r', 'U_r', 'S_k']
    assert 'data' not in variables
    assert 'r' not in variables
    variables['data'] = raw_props
    variables['r'] = r

    from copy import deepcopy
    temp_vars = deepcopy(variables)
    temp_vars['LR_parms'] = A
    data = np.empty(shape=(*numbers_of_values, len(raw_props), len(r)))
    data[:] = np.nan
    raw_data = xr.DataArray(data=data,
                            dims=tuple(temp_vars.keys()),
                            coords=temp_vars)
    del variables['data']
    del variables['r']
    return thermo_data, raw_data


def format_and_save_figures(fig_gr, ax_gr,
                            fig_ur, ax_ur,
                            fig_sk, ax_sk,
                            xlim=(0, 5),
                            **kwargs):
    ur_ylim = kwargs.get('ur_ylim')

    ax_gr.set_xlabel('r')
    ax_gr.set_ylabel('g(r)')
    if xlim:
        ax_gr.set_xlim((0, 5))
    ax_gr.legend(loc='upper right')
    fig_gr.savefig('figs/g_r.pdf', bbox_inches='tight')

    ax_ur.set_xlabel('r')
    ax_ur.set_ylabel('U(r)')
    if ur_ylim:
        ax_ur.set_ylim(ur_ylim)
    if xlim:
        ax_ur.set_xlim((0, 5))
    ax_ur.legend(loc='upper right')
    fig_ur.savefig('figs/U_r.pdf', bbox_inches='tight')

    ax_sk.set_xlabel('k')
    ax_sk.set_ylabel('S(k)')
    ax_sk.legend(loc='upper right')
    ax_sk.set_xlim((0, 50))
    fig_sk.savefig('figs/S_k.pdf', bbox_inches='tight')

