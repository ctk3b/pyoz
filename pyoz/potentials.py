from collections import OrderedDict
import inspect
import itertools as it

import numpy as np
import pandas as pd

from pyoz.exceptions import PyozError
import pyoz.unit as u
from pyoz.unit import AVOGADRO_CONSTANT_NA as Na
from pyoz.unit import BOLTZMANN_CONSTANT_kB as kB


def arithmetic(a, b):
    return 0.5 * (a + b)


def geometric(a, b):
    return np.sqrt(a * b)

mixing_functions = {'arithmetic': arithmetic,
                    'geometric': geometric}


class TotalPotential(object):
    """Container class for the total potential, U_ij(r).

    Parameters
    ----------
    r : np.ndarray, shape=(n_points,), dtype=float
        The radii at which the potentials are evaluated.
    n_components : int
        The number of components in the system.
    potentials : set of ContinuousPotential
        All potentials that make up the total potential of the system.

    Attributes
    ----------
    ij : np.ndarray, shape=(n_comps, n_comps, n_points), dtype=float
        The total U_ij potential.

    """
    def __init__(self, r, n_components, potentials):
        matrix_shape = (n_components, n_components, r.shape[0])
        self.ij = np.zeros(shape=matrix_shape)

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
                descr.append(' + ')
        descr.append('>')
        return ''.join(descr)


class Potential(object):
    """Base-class for all potentials.

    TODO: See what else can be abstracted away from the sub-classes
    """
    def __repr__(self):
        return self.__class__.__name__


class ContinuousPotential(Potential):
    """A continuous pair potential.

    * We always assume that `r` is the first parameter in `potential_func`
    * When no mixing rule or cross interaction is specified for a pair, they
        do not interact.

    """
    def __init__(self, system, potential_func, **mixing_rules):
        super().__init__()
        self.system = system
        self.potential_func = potential_func
        # Assume that `r` is the first parameter in `potential_func`.
        self.parameter_names = [p for p in
                                inspect.signature(potential_func).parameters][1:]
        self.n_parameters = len(self.parameter_names)

        # TODO: revisit using pd.Panel()
        self.parameters = pd.DataFrame(columns=self.parameter_names)
        self.parm_ij = np.zeros(shape=(self.n_parameters, 1, 1))
        self.ij = None

        if len(mixing_rules) > self.n_parameters:
            raise PyozError('Provided more mixing rules than there are '
                            'parameters.')
        for rule in mixing_rules.keys():
            if rule not in self.parameters.columns:
                valid_parms = [p for p in sorted(self.parameters.columns)]
                raise PyozError('Provided mixing rule for unknown parameter: '
                                '{}. Valid parameters are "{}"'.format(
                    rule, valid_parms)
                )

        # TODO: robust getters/setters for changing mixing rules
        self.mixing_rules = mixing_rules
        self._mixing_funcs = dict()
        for parm in self.parameters.columns:
            func_name = mixing_rules.get(parm)
            if func_name:
                func = mixing_functions[func_name]
            else:
                func = None
            self._mixing_funcs[parm] = func

    @property
    def n_components(self):
        return len(self.parameters.index)

    def add_component(self, component, **parameters):
        wrong_num = len(parameters) != self.n_parameters
        wrong_parms = not all(p in self.parameters.columns
                              for p in parameters.keys())
        if wrong_num or wrong_parms:
            provided_parms = [p for p in sorted(parameters.keys())]
            required_parms = [p for p in sorted(self.parameters.columns)]
            raise PyozError('Tried to add component with parameters "{}".'
                            'Potential requires parameters "{}".'.format(
                provided_parms, required_parms)
            )
        reduced_parameters = self._reduce_units(parameters)
        self.parameters.loc[component] = reduced_parameters

        # TODO: less disguting method to expand this array
        if self.n_components > 1:
            self._expand_parm_ij()

        component_idx = self._component_idx(component)
        for parm, value in reduced_parameters.items():
            parm_idx = self._parameter_idx(parm)
            self.parm_ij[parm_idx, component_idx, component_idx] = value
            if self.mixing_rules and self.n_components > 1:
                self._apply_mixing_rules(parm, parm_idx)

        component.potentials.add(self)

    def _reduce_units(self, parameters):
        # TODO: robust handling for units that have both an energy
        # and other components
        reduced_parameters = dict()
        for parm_name, parm in tuple(parameters.items()):
            try:  # Are we dealing with energy unit?
                in_kJ_per_mol = parm.in_units_of(u.kilojoules_per_mole)
            except TypeError:
                unitless = parm.value_in_unit_system(u.pyoz_unit_system)
            except AttributeError:
                unitless = parm
            else:  # It's an energy unit.
                unitless = in_kJ_per_mol / (Na * kB * self.system.T)
            reduced_parameters[parm_name] = unitless
        return reduced_parameters

    def _expand_parm_ij(self):
        old_ij = np.copy(self.parm_ij)
        self.parm_ij = np.zeros(shape=(self.n_parameters,
                                       self.n_components,
                                       self.n_components))
        for n, i, j in np.ndindex(old_ij.shape):
            self.parm_ij[n, i, j] = old_ij[n, i, j]

    def _apply_mixing_rules(self, parm, parm_idx):
        mixer = self._mixing_funcs[parm]
        if not mixer:
            return
        for i, j in it.permutations(range(self.n_components), 2):
            parm_i = self.parameters[parm][i]
            parm_j = self.parameters[parm][j]
            self.parm_ij[parm_idx, i, j] = mixer(parm_i, parm_j)

    def _component_idx(self, component):
        return self.parameters.index.get_loc(component)

    def _parameter_idx(self, parameter):
        return self.parameters.columns.get_loc(parameter)

    def add_binary_interaction(self, comp1, comp2, **parameters):
        # TODO: Do we ever need asymmetric interactions? U(c1, c2) != U(c2, c1)
        if any(c not in self.parameters.index for c in [comp1, comp2]):
            raise PyozError('Components must be added to the potential before '
                            'adding a binary interaction between them.')

        reduced_parameters = self._reduce_units(parameters)
        comp1_idx = self._component_idx(comp1)
        comp2_idx = self._component_idx(comp2)
        for parm, value in reduced_parameters.items():
            parm_idx = self._parameter_idx(parm)
            self.parm_ij[parm_idx, comp1_idx, comp2_idx] = value
            self.parm_ij[parm_idx, comp2_idx, comp1_idx] = value

    def apply(self):
        n_components = self.n_components
        r = self.system.r

        self.ij = np.zeros(shape=(n_components, n_components, r.shape[0]))
        for i, j in np.ndindex(self.parm_ij.shape[1:]):
            p = self.parm_ij[:, i, j]
            self.ij[i, j, :] = self.potential_func(r, *p)


class LennardJones(ContinuousPotential):
    def __init__(self, system, **mixing_rules):
        def lj_func(r, eps, sig):
            return 4 * eps * ((sig / r)**12 - (sig / r)**6)
        super().__init__(system=system, potential_func=lj_func, **mixing_rules)


class Coulomb(ContinuousPotential):
    def __init__(self, system):
        def coulomb(r, q):
            return q**2 / r
        super().__init__(system=system, potential_func=coulomb, q='geometric')

# ========================================================= #
# Explicit potential classes, primarily for testing purposes.
# ========================================================= #


class _LennardJones(object):
    def __init__(self, system, sig_rule='arithmetic', eps_rule='geometric'):
        self.system = system
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

    def add_component(self, component, sig, eps):
        # TODO: robust unit checking
        self.sig[component] = sig
        self.eps[component] = eps

    def apply(self):
        r, T = self.system.r, self.system.T
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
