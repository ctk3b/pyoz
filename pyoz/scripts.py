def cmdline_app(args=None):
    import argparse
    import logging
    import os
    import sys
    from time import time

    import numpy as np
    from numpy import array, mat, copy, eye, fromfile, zeros, isfinite, empty
    import yaml

    import pyoz
    from pyoz import settings
    from pyoz.potential import Potential, MayerF
    import pyoz.thermodynamic_properties as properties

    from pyoz import dft as ft
    from pyoz.closure import calcGammaTerm
    from pyoz.misc import convergence_dsqn, dotproduct



    if not args:
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--version',
                            action='version',
                            version='{:s}'.format(pyoz.__version__))
        parser.add_argument('-i', '--input',
                            dest='script',
                            default='',
                            metavar='FILE',
                            help='YAML configuration script specifying input'
                                 ' parameters.')
        parser.add_argument('-l', '--logfile',
                            dest='logfile',
                            default='pyoz.log',
                            metavar='FILE',
                            help='Location of output log file.')
        parser.add_argument('-g', '--gamma',
                            dest='gamma',
                            default=None,
                            metavar='GAMMA',
                            help='Initial values of gamma function.')
        args = vars(parser.parse_args())

    if not args['script']:
        raise ValueError('No input script provided.')

    logging.basicConfig(filename=args['logfile'])
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info('pyOZ - version {:s}'.format(pyoz.__version__))

    with open(args['script']) as yaml_file:
        script = yaml.load(yaml_file)
    settings.update(script)

    dr = settings['dr']
    dk = settings['dk']
    n_points = settings['n_points']
    r = np.linspace(dr, n_points * dr - dr, n_points - 1)
    k = np.linspace(dk, n_points * dk - dk, n_points - 1)

    # U = Potential(r=r, n_components=settings['n_components'],
    #               potentials=settings['potentials'])

    lj_parms = settings['potentials']['lennard-jones']
    lj_parms['sigma'] = np.array([float(x) for x in lj_parms['sigma'].split()])
    lj_parms['epsilon'] = np.array([float(x) for x in lj_parms['epsilon'].split()])

    U = Potential(r, n_components=2, potentials=settings['potentials'])
    import ipdb; ipdb.set_trace()

    U_erf_ij_real = np.zeros_like(r)
    U_erf_ij_fourier = np.zeros_like(r)
    M = np.exp(-U_ij) - 1
    mod_M = M + 1
    mod_M_erf = np.exp(U_erf_ij_real)
    G_r_ij = -U_erf_ij_real

    def percus_yevick(mod_M, mod_M_erf, G_r_ij):
        gamma = 1 + G_r_ij
        g_r_ij = mod_M * mod_M_erf * gamma
        c_r_ij = g_r_ij - G_r_ij - 1
        return c_r_ij, g_r_ij

    converged = False
    total_iter = 0
    n_iter = 0

    dft = ft.dft(n_points, dr, dk, r, k)
    dft.print_status()

    # mol/L to particle / A^3
    density = 0.5 * 6.0221415 * 1e-4
    while not converged and n_iter < 100:
        n_iter += 1
        total_iter += 1
        cs_r_ij, g_r_ij = percus_yevick(mod_M, mod_M_erf, G_r_ij)
        Cs_f_ij, C_f_ij = dft.dfbt(cs_r_ij, norm=density, corr=U_erf_ij_fourier)

        from numpy import linalg
        H_f_ij = linalg.solve((1 - C_f_ij), C_f_ij, n_points)

        S = 1 + H_f_ij
        G_f_ij = S - 1 - Cs_f_ij

        G_n_ij = dft.idfbt(G_f_ij, norm=density, corr=U_erf_ij_real)
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    cmdline_app()
