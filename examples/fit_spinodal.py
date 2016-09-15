import itertools as it
import pickle
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import leastsq, least_squares
import xarray as xr

from phase_scan_SALR import number_densities

sns.set_style('whitegrid', {'xtick.major.size': 5,
                            'ytick.major.size': 5,
                            'axes.edgecolor': 'k',
                            'font.weight': 'bold'})

def residual(params, rho, sk0, branch):
    sk0_pred = sk0_powerlaw(params, rho, branch)
    return (sk0_pred/sk0 - 1.0)**2  # relative error.


def sk0_powerlaw(params, rhos, branch='vap'):
    c, rhostar, exponent = params
    if branch == 'liq':
        sk0 = c / (rhos - rhostar)**exponent
    elif branch == 'vap':
        sk0 = c / (rhostar - rhos)**exponent
    else:
        raise Exception('Branch should be `vap` or `liq`')
    return sk0


def fit_sk0_powerlaw(guesses, rhos, sk0, branch):
    # params_opt, pcov = leastsq(func=residual,
    #                            x0=guesses,
    #                            args=(rhos, sk0, branch))
    # return params_opt
    if branch == 'vap':
        bounds = [(0.01, 0, 0.1),
                  (20, 0.5, 5)]
    elif branch == 'liq':
        bounds = [(0.01, 0.2, 0.1),
                  (20, 0.8, 5)]
    results = least_squares(fun=residual,
                            x0=guesses,
                            bounds=bounds,
                            args=(rhos, sk0, branch))
    return results.x


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_vap_liq_split(sk0):
    window_size = 10
    bool_indices = np.all(rolling_window(sk0.to_masked_array().mask, window_size), axis=1)
    indices = np.mgrid[0:len(bool_indices)][bool_indices]

    windows_in_a_row = len(indices)
    if windows_in_a_row >= 1:
        last_vap = indices[0]
        return last_vap
    return None


if __name__ == '__main__':
    # do_fit = False
    do_fit = True
    prefix = 'uncharged_'

    # thermo_data = xr.open_dataset('data/test_exp_thermo_data.nc')['THERMO']
    thermo_data = xr.open_dataset('data/{}thermo_data.nc'.format(prefix))['THERMO']
    # For some reason the data-type isn't preserved and loads back as '|S8'
    thermo_data.coords['thermo'] = thermo_data.coords['thermo'].astype('<U8')
    thermo_data.coords['mn'] = thermo_data.coords['mn'].astype('<U8')

    fig_fit, ax_fit = plt.subplots()
    fig_sp, ax_sp = plt.subplots()

    def rho_from_wt(sio2_wt):
        return 6 / np.pi * (sio2_wt / 100) / 2.65
    sio2 = thermo_data.coords['SiO2']
    rhos = np.array([rho_from_wt(x) for x in sio2])

    epsilons = thermo_data.coords['epsilon']
    # plot_only = [0.40, 0.42, 0.50, 0.60]
    palette = sns.color_palette()


    mn_pairs = thermo_data.coords['mn']
    shapes = ['o', 'v', 'D', '^', 's']


    marker = {mn.item(): shape for mn, shape in zip(mn_pairs, shapes)}
    if do_fit:
        n_colors = len(mn_pairs)
        palette = sns.color_palette('deep',
                                    n_colors=n_colors)
        colors = {mn.item(): color for mn, color in zip(mn_pairs, palette)}
    else:
        n_colors = len(epsilons)
        palette = sns.cubehelix_palette(n_colors, start=.5, rot=-.75, reverse=True)
        palette = sns.color_palette(n_colors=n_colors)
        colors = {eps.item(): color for eps, color in zip(epsilons, palette)}

    import pandas as pd
    spin_vap = pd.DataFrame(index=mn_pairs.values,
                            columns=[round(x, 3) for x in epsilons.values])
    spin_liq = pd.DataFrame(index=mn_pairs.values,
                            columns=[round(x, 3) for x in epsilons.values])

    for mn, eps in it.product(mn_pairs, epsilons):
        # if round(eps.item(), 2) not in plot_only:
        #     continue
        fmt_state = '{} {:.3f}'.format(mn.item(), eps.item())

        sk0 = thermo_data.sel(mn=mn, epsilon=eps, thermo='Sk0')

        if do_fit:
            if np.mean(sk0) < 1.0:
                continue
            last_vap = find_vap_liq_split(sk0)
            if last_vap is None:
                # print('No phase split for ', fmt_state)
                continue
            sk0_vap = sk0[:last_vap]
            not_nan_indices = ~np.isnan(sk0_vap).values
            sk0_vap = sk0_vap[not_nan_indices]
            rho_vap = rhos[:last_vap]
            rho_vap = rho_vap[not_nan_indices]

            sk0_liq = sk0[last_vap:]
            not_nan_indices = ~np.isnan(sk0_liq).values
            sk0_liq = sk0_liq[not_nan_indices]
            rho_liq = rhos[last_vap:]
            rho_liq = rho_liq[not_nan_indices]
            if len(sk0_vap) < 3 or len(sk0_liq) < 3:
                # print('Not enough data for ', fmt_state)
                continue
        shape = marker[mn.item()]
        if do_fit:
            color = colors[mn.item()]
        else:
            color = colors[eps.item()]

        dots = ax_fit.plot(rhos, sk0,
                           lw=0.5,
                           color=color,
                           marker=shape,
                           label=fmt_state)

        if do_fit:
            branch = 'vap'
            vap_guesses = (1.0, np.max(rho_vap)*1.25, 1.5)
            (c, rho_sp, exponent) = fit_sk0_powerlaw(vap_guesses,
                                                     rho_vap,
                                                     sk0_vap,
                                                     branch=branch)

            sk0_vap_fit = sk0_powerlaw((c, rho_sp, exponent), rho_vap, branch)
            ax_fit.plot(rho_vap, sk0_vap_fit, color=dots[0].get_color())
            spin_vap.loc[mn.item(), round(eps.item(), 3)] = rho_sp

            print(fmt_state, 'vap', rho_sp, c, exponent)
            branch = 'liq'
            liq_guesses = (1.0, np.min(rho_liq)-0.05, 1.5)
            (c, rho_sp, exponent) = fit_sk0_powerlaw(liq_guesses,
                                                     rho_liq,
                                                     sk0_liq,
                                                     branch=branch)

            sk0_liq_fit = sk0_powerlaw((c, rho_sp, exponent), rho_liq, branch)
            ax_fit.plot(rho_liq, sk0_liq_fit, color=dots[0].get_color())
            spin_liq.loc[mn.item(), round(eps.item(), 3)] = rho_sp
            # print(fmt_state, 'liq', rho_sp)

    ax_fit.legend(loc='upper left')
    ax_fit.set_xlabel(r'$\rho$')
    ax_fit.set_ylabel('S(k=0)')
    ax_fit.legend(bbox_to_anchor=(1.3, 1.0))
    # ax_fit.set_ylim(0, 100)

    sns.despine(fig_fit, ax_fit)
    if do_fit:
        fig_fit.savefig('sk0_{}fit.pdf'.format(prefix), bbox_inches='tight')
    else:
        fig_fit.savefig('sk0_{}all.pdf'.format(prefix), bbox_inches='tight')
    # import ipdb; ipdb.set_trace()


    if do_fit:
        for mn in mn_pairs:
            color = colors[mn.item()]
            ax_sp.plot(spin_vap.loc[mn.item()], epsilons,
                       marker='o', color=color, label='vap ' + mn.item())
            ax_sp.plot(spin_liq.loc[mn.item()], epsilons,
                       marker='v', color=color, label='liq ' + mn.item())

        ax_sp.set_xlabel(r'$\rho$')
        ax_sp.set_ylabel(r'$\epsilon$')
        sns.despine(fig_sp, ax_sp)
        ax_sp.legend(bbox_to_anchor=(1.3, 1.0))
        fig_sp.savefig('{}spinodal.pdf'.format(prefix), bbox_inches='tight')


    # import ipdb; ipdb.set_trace()


    # fig, ax = plt.subplots()
    # for x in range(1, 8):
    #     m = 12*x
    #     es, rho_star = zip(*spinodal[m])
    #     ax.plot(rho_star, [1/x for x in es], label=m)
    # ax.legend(bbox_to_anchor=(1.3, 1.0))
    # fig.savefig('spinodal.pdf', bbox_inches='tight')
    #
    # import ipdb; ipdb.set_trace()




    # with open('all_data.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    # d_SiO2_wt_perc = 1
    # SiO2_wt_perc = np.arange(5, 50, d_SiO2_wt_perc)
    # rhos = np.array([number_densities(wt) for wt in SiO2_wt_perc])
    #
    # eps = np.arange(1.0, 1.5, 0.05)

    # from collections import OrderedDict
    # spinodal = OrderedDict()
    # markers = ['o', 'v', '^', '>', '8', 'D', 'H', 'x']
    # for x in range(1, 8):
    #     m = x * 12
    #     spinodal[m] = list()
    #     for e in eps:
    #         sk0 = get_sk0(e, m, data)
    #         n_data_points = len(sk0[~np.isnan(sk0)])
    #         if all(np.isnan(x) for x in sk0) or n_data_points < 3:
    #             continue
    #         print('fitting eps={:.3f}'.format(e))
    #         guesses = (1.0, np.max(rhos)*1.25, 1.5)
    #         (c, rho_sp, exponent), pcov = fit_sk0_powerlaw(guesses, rhos, sk0, branch='left')
    #
    #         spinodal[m].append((e, rho_sp))
    #         dots = ax_fit.plot(rhos, sk0, lw=0, marker=markers[int(m/12 -1)], label='{:.2f} {}'.format(e, m))
    #         sk0_fit = sk0_powerlaw((c, rho_sp, exponent), rhos, 'left')
    #         ax_fit.plot(rhos, sk0_fit, color=dots[0].get_color())
    #         # import ipdb; ipdb.set_trace()

