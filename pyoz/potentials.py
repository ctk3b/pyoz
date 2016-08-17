from collections import OrderedDict

import numpy as np
import simtk.unit as u
from simtk.unit import AVOGADRO_CONSTANT_NA as Na
from simtk.unit import BOLTZMANN_CONSTANT_kB as kB

from pyoz.exceptions import PyozError


def arithmetic(a, b):
    return 0.5 * (a + b)


def geometric(a, b):
    return np.sqrt(a * b)

mixing_functions = {'arithmetic': arithmetic,
                    'geometric': geometric}


class TotalPotential(object):
    """Calculate the total U_ij potential.

    Also stores:
    * individual potentials which sum to the total interaction
    * discontinuities of the individual potentials

    Currently supported potentials:
    * lennard-jones

    Parameters
    ----------
    r : np.ndarray, shape=(n_points,), dtype=float
    n_components : int
    potentials : dict

    Attributes
    ----------
    ij : np.ndarray, shape=(n_comps, n_comps, n_points), dtype=float
        The total U_ij potential.
    ij_ind : np.ndarray, shape=(n_pots, n_comps, n_comps, n_points), dtype=float
        The individual contributions to the total potential.

    """
    def __init__(self, r, n_components, potentials):
        n_potentials = len(potentials)
        matrix_shape = (n_components, n_components, r.shape[0])
        self.ij = np.zeros(shape=matrix_shape)
        self.ij_ind = np.zeros(shape=(n_potentials,
                                      n_components, n_components, r.shape[0]))

        self.erf_real = np.zeros(shape=matrix_shape)
        self.erf_fourier = np.zeros(shape=matrix_shape)

        for potential in potentials:
            self.ij += potential.ij
        self.potentials = potentials

    def __repr__(self):
        descr = list('<Total potential: ')
        n_potentials = len(self.potentials)
        for n, pot in enumerate(self.potentials):
            descr.append('{}'.format(pot))
            if n < n_potentials - 1:
                descr.append('+ ')
        descr.append('>')
        return ''.join(descr)


class ContinuousPotential(object):
    def __init__(self, potential_function, **mixing_rules):
        self.potential_function = potential_function
        # TODO: properly implement pd.DataFrame to handle parameters
        # TODO: robust getters/setters for changing mixing rules
        # Currently, "robustness" is provided simply by alphabetic ordering.
        self.mixing_rules = mixing_rules
        self._mixing_funcs = [mixing_functions[rule]
                              for parameter, rule in sorted(mixing_rules.items())]
        self.parameters = OrderedDict()

        self.ij = None

    @property
    def n_parameter_types(self):
        return len(list(self.parameters.values())[0])

    @property
    def n_components(self):
        return len(self.parameters)

    def add_parameters(self, component, **parameters):
        # Maintain identical order of parameters for each component.
        self.parameters[component] = OrderedDict(sorted(parameters.items()))

    def apply(self, r, T):
        n_components = self.n_components

        parms = np.empty(shape=(self.n_parameter_types, n_components))
        for comp_n, comp_parms in enumerate(self.parameters.values()):
            for parm_n, parm in enumerate(comp_parms.values()):
                parms[parm_n, comp_n] = parm

        P_ij = np.empty(shape=(self.n_parameter_types, n_components, n_components))
        for n, i, j in np.ndindex(P_ij.shape):
            mixer = self._mixing_funcs[n]
            P_ij[n, i, j] = mixer(parms[n, i], parms[n, j])

        self.ij = np.zeros(shape=(n_components, n_components, r.shape[0]))
        for i, j in np.ndindex(P_ij.shape[1:]):
            p = P_ij[:, i, j]
            self.ij[i, j, :] = self.potential_function(r, *p)


class LennardJones(ContinuousPotential):
    def __init__(self, **mixing_rules):
        def lj_func(r, e, s):
            return 4 * e * ((s / r)**12 - (s / r)**6)
        super().__init__(lj_func, **mixing_rules)

# ========================================================= #
# Explicit potential classes, primarily for testing purposes.
# ========================================================= #


class _LennardJones(object):
    def __init__(self, sig_rule='arithmetic', eps_rule='geometric'):
        self.sig = OrderedDict()
        self.eps = OrderedDict()
        self._sig_rule = sig_rule
        self._eps_rule = eps_rule
        self._mix_sig = mixing_functions[sig_rule]
        self._mix_eps = mixing_functions[eps_rule]

        self.sig_ij = None
        self.eps_ij = None
        self.ij = None

    @property
    def sig_rule(self):
        return self._sig_rule

    @sig_rule.setter
    def sig_rule(self, rule):
        self._sig_rule = rule
        self._mix_sig = mixing_functions[rule]

    @property
    def eps_rule(self):
        return self._eps_rule

    @eps_rule.setter
    def eps_rule(self, rule):
        self._eps_rule = rule
        self._mix_eps = mixing_functions[rule]

    def add_parameters(self, component, sig, eps):
        # TODO: robust unit checking
        self.sig[component] = sig
        self.eps[component] = eps

    def apply(self, r, T):
        sig = np.array([x.value_in_unit(u.angstroms)
                        for x in self.sig.values()])
        eps = np.array([x / Na / kB / T
                        for x in self.eps.values()])

        n_components = len(sig)
        self.sig_ij = np.zeros(shape=(n_components, n_components))
        self.eps_ij = np.zeros(shape=(n_components, n_components))
        self.ij = np.zeros(shape=(n_components, n_components, r.shape[0]))
        for i, j in np.ndindex(self.sig_ij.shape):
            s = self._mix_sig(sig[i], sig[j])
            e = self._mix_eps(eps[i], eps[j])
            self.sig_ij[i, j] = s
            self.eps_ij[i, j] = e
            self.ij[i, j, :] = 4 * e * ((s / r)**12 - (s / r)**6)

    def __repr__(self):
        return self.__class__.__name__
