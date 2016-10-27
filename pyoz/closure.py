import numpy as np


def hypernetted_chain(U_r, e_r, kT, **kwargs):
    """Apply the hyper-netted chains closure.

    .. math::
        g(r) = exp(-\Beta U(r) + e(r))

    """
    c_r = np.exp(-U_r / kT + e_r) - e_r - 1
    return c_r


def reference_hypernetted_chain(U_r, e_r, kT, **kwargs):
    """Apply the reference hyper-netted chains closure.

    .. math::
        \Delta U = U(r) - U(r)_{ref}

        \Delta e = e(r) - e(r)_{ref}

        g(r) = g(r)_{ref} * exp(-\Beta \Delta U + \Delta e)

    """
    ref_system = kwargs['reference_system']
    g_r_ref = ref_system.g_r
    e_r_ref = ref_system.e_r
    U_r_ref = ref_system.U_r

    dU = U_r - U_r_ref
    de = e_r - e_r_ref
    c_r = g_r_ref * np.exp(-dU / kT + de) - e_r - 1
    return c_r


def percus_yevick(U_r, e_r, kT, **kwargs):
    """Apply the Percus-Yevick closure.

    .. math::
        g(r) = exp(-\Beta U(r)) * (1 + e(r))

    """
    c_r = np.exp(-U_r / kT) * (1 + e_r) - e_r - 1
    return c_r


def kovalenko_hirata(U_r, e_r, kT,  **kwargs):
    """Apply the Kovalenko-Hirata closure.

    .. math::
        d(r) = -\Beta U(r) + e(r)

        g(r) &= exp(d(r))    for d(r) <= 0 \\
             &= 1 + d(r)     for d(r) > 0

    References
    ----------
    .. [1] http://dx.doi.org/10.1063/1.481676

    """
    beta_U = -U_r / kT
    d_r = beta_U + e_r
    hnc_c_r = np.exp(d_r) - e_r - 1
    c_r = np.where(d_r <= 0, hnc_c_r, beta_U)
    return c_r


def duh_henderson(U_r, e_r, kT,  **kwargs):
    """Apply the Duh-Henderson closure.

    .. math::
        B(r) =

        g(r) = exp(-\Beta U(r) + e(r) + B(r))


    References
    ----------
    .. [1] http://dx.doi.org/10.1063/1.471391

    """

    s_r = e_r - U_r / kT
    B_r = np.where(s_r > 0,
                   -0.5 * s_r**2 * (9 + 7 * s_r) / (3 + s_r) / (3 + 5 * s_r),
                   -0.5 * s_r **2)
    c_r = np.exp(-U_r / kT + e_r + B_r) - e_r - 1
    return c_r


supported_closures = {'hnc': hypernetted_chain,
                      'hypernetted chain': hypernetted_chain,
                      'hyper-netted chain': hypernetted_chain,
                      'hypernetted-chain': hypernetted_chain,

                      'rhnc': reference_hypernetted_chain,
                      'reference hypernetted chain': reference_hypernetted_chain,
                      'reference hyper-netted chain': reference_hypernetted_chain,
                      'reference hypernetted-chain': reference_hypernetted_chain,

                      'py': percus_yevick,
                      'percus yevick': percus_yevick,
                      'percus-yevick': percus_yevick,

                      'kh': kovalenko_hirata,
                      'kovalenko-hirata': kovalenko_hirata,
                      'kovalenko hirata': kovalenko_hirata,

                      'dh': duh_henderson,
                      'duh-henderson': duh_henderson,
                      'duh henderson': duh_henderson,
}
closure_names = supported_closures.keys()


# Currently unimplemented closures on the wishlist.
def partial_series_expansion_n(U_r, e_r, kT,  **kwargs):
    # See https://github.com/ctk3b/pyoz/issues/21
    pass




def scoza(U_r, e_r, kT,  **kwargs):
    # See https://github.com/ctk3b/pyoz/issues/22
    pass

