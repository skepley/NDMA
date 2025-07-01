import numpy as np
import itertools

from ndma.basic_models.EMT_model import def_emt_hill_model
from ndma.bifurcation.saddlenode import SaddleNode


f = def_emt_hill_model()


def Hmin(n, par, x):
    ell, delta, theta = par[:]
    return ell + delta / (1 + (x / theta) ** n)


def H_bound():
    ell, delta, theta = 2.1, 3.4, 1.6
    return np.array((ell, ell + delta))


def Hplus(n, par, x):
    ell, delta, theta = par[:]
    if x == 0:
        return ell
    return ell + delta / (1 + (theta / x) ** n)


# the EMT right hand side

def gpm_func(n, par):
    # change this, the NDMA parameters are gamma1, ell21, delta21, theta21, ell31, delta31....,gamma2....
    if not hasattr(n, "__len__"):
        n = n + np.zeros(12)
    gplus = lambda x: np.array((1,
                                1,
                                Hplus(n[4], par[4, :], x[0]),
                                1,
                                Hplus(n[8], par[8, :], x[2]),
                                1))

    gminus = lambda x: np.array((Hmin(n[0], par[0, :], x[1]) * Hmin(n[1], par[1, :], x[3]),
                                 Hmin(n[2], par[2, :], x[2]) * Hmin(n[3], par[3, :], x[4]),
                                 Hmin(n[5], par[5, :], x[5]),
                                 Hmin(n[6], par[6, :], x[4]),
                                 Hmin(n[7], par[7, :], x[1]) * Hmin(n[9], par[9, :], x[3]),
                                 Hmin(n[10], par[10, :], x[2]) * Hmin(n[11], par[11, :], x[4])))
    return gplus, gminus


def F_func(n, par, gamma):
    gplus, gminus = gpm_func(n, par)
    F = lambda x: - gamma * x + gplus(x) * gminus(x)
    return F


def phi_func(n, par, gamma):
    gplus, gminus = gpm_func(n, par)
    phi = lambda xplus, xminus: (gplus(xplus) * gminus(xminus) / gamma, gplus(xminus) * gminus(xplus) / gamma)
    return phi


def is_degenerate(xminus, xplus, tol=10**-7):
    if np.linalg.norm(xminus - xplus) < tol:
        return True
    return False


def convergence(F, xminus, xplus, tol=10 ** -7):
    if is_degenerate(xminus, xplus, tol):
        return True
    zero_corners = 0
    allx = list(itertools.product(*zip(xminus, xplus)))

    for i in range(2 ** 6):
        if (np.linalg.norm(F(allx[i]))) < tol:
            zero_corners += 1

    if zero_corners >= 2:
        return True
    else:
        return False


def boxy_box_from_pars(n, par, gamma, maxiter=2000):
    # define the mapping

    gplus, gminus = gpm_func(n, par)
    phi = phi_func(n, par, gamma)
    F = F_func(n, par, gamma)

    # set starting point
    xzero = np.zeros(6)
    x100 = 100 + xzero
    xplus, xminus = gplus(xzero) * gminus(x100) / gamma, gplus(x100) * gminus(xzero) / gamma

    # the iterations
    iter = 0
    remainder = np.array([])
    new_remainder = 10
    while new_remainder > 10**-7 and (iter < maxiter):
        xplus_new, xminus_new = phi(xplus, xminus)
        new_remainder = np.linalg.norm(xplus - xplus_new) + np.linalg.norm(xminus_new - xminus)
        remainder = np.append(remainder, new_remainder)
        iter += 1
        xplus, xminus = xplus_new, xminus_new

    # wrapping of results
    if iter == maxiter:
        success = False
    else:
        success = True

    return success, xminus, xplus, remainder


def NDMApars_to_boxyboxpars(hill, pars):
    # the NDMA pars are all mixed up!
    gamma_index = [0, 7, 14, 21, 25, 35]
    gamma = pars[gamma_index]
    par = np.delete(pars, gamma_index)
    par = np.reshape(par, [12, 3])
    return hill, par, gamma


def NDMAparsTHETA_to_boxyboxpars(theta, pars43, theta_index):
    # the NDMA pars are all mixed up!
    pars43[theta_index] = theta
    hill_index = 0
    hill = pars43[hill_index]
    pars42 = np.delete(pars43, hill_index)
    gamma_index = [0, 7, 14, 21, 25, 35]
    gamma = pars42[gamma_index]
    par = np.delete(pars42, gamma_index)
    par = np.reshape(par, [12, 3])
    return hill, par, gamma


def corners_of_box(xminus, xplus):
    all_corners = list(itertools.product(*zip(xminus, xplus)))
    return np.array(all_corners)


def approx_saddle_node_with_boxy_box(hill_comb, par_NDMA):
    """ by going through all hill coefficients stored in hill_comb, we look for changes in the number of equilibria and
    return approximate saddle nodes"""
    def outlier(xminus, xplus, hill_iter, old_xminus, old_xplus, old_hill, tol=10**-7):
        """ select the corner that is disappearing during the saddle node"""
        if is_degenerate(old_xminus, old_xplus, tol=tol):
            return outlier(old_xminus, old_xplus, old_hill, xminus, xplus, hill_iter, tol=tol)
        degenerate_x = xminus
        hill = hill_iter
        approx_saddle_position = 0 * degenerate_x
        for i in range(np.size(degenerate_x)):
            if np.abs(degenerate_x[i] - old_xminus[i]) > np.abs(degenerate_x[i] - old_xplus[i]):
                approx_saddle_position[i] = old_xminus[i]
            else:
                approx_saddle_position[i] = old_xplus[i]
        return approx_saddle_position, hill

    if np.size(hill_comb) == 2:
        low_hill = min(hill_comb)
        high_hill = max(hill_comb)
        hill_comb = np.linspace(high_hill, low_hill, 50)

    old_hill, par, gamma = NDMApars_to_boxyboxpars(hill_comb[0], par_NDMA)
    success, old_xminus, old_xplus, remainder = boxy_box_from_pars(old_hill, par, gamma, maxiter=300)
    approx_saddle_position, approx_saddle_hill = [], []
    for hill_iter in hill_comb[1:]:
        success, xminus, xplus, remainder = boxy_box_from_pars(hill_iter, par, gamma, maxiter=300)
        if not success:
            continue
        if is_degenerate(xminus, xplus, tol=10**-5) != is_degenerate(old_xminus, old_xplus, tol=10**-5):
            coord, hill_val = outlier(xminus, xplus, hill_iter, old_xminus, old_xplus, old_hill, tol=10**-5)
            approx_saddle_position.append(coord)
            approx_saddle_hill.append(hill_val)
        old_xminus, old_xplus, old_hill = xminus, xplus, hill_iter

    return approx_saddle_position, approx_saddle_hill


def saddle_node_with_boxybox(saddle_node_problem, hill_comb, par_NDMA):
    """ for the EMT saddle node problem taken a selection of hill coefficients and a parameter, it finds all saddle nodes"""
    if np.size(hill_comb) == 2:
        low_hill = min(hill_comb)
        high_hill = max(hill_comb)
        hill_comb = np.linspace(high_hill, low_hill, 50)
    approx_saddle_position, old_hill = approx_saddle_node_with_boxy_box(hill_comb, par_NDMA)
    par_of_SNbif, bad_candidate = [], []
    for i in range(len(old_hill)):
        saddle = saddle_node_problem.find_saddle_node(0, old_hill[i], par_NDMA, equilibria=approx_saddle_position[i])
        if saddle:
            par_of_SNbif.append(saddle)
        else:
            bad_candidate.append(old_hill[i])
    return par_of_SNbif, bad_candidate


def approx_saddle_node_with_boxy_box_THETA(theta_comb, par_NDMA, index_to_varry):
    """ by going through all hill coefficients stored in hill_comb, we look for changes in the number of equilibria and
    return approximate saddle nodes"""
    def outlier(xminus, xplus, theta_iter, old_xminus, old_xplus, old_theta, tol=10**-7):
        """ select the corner that is disappearing during the saddle node"""
        if is_degenerate(old_xminus, old_xplus, tol=tol):
            return outlier(old_xminus, old_xplus, old_theta, xminus, xplus, theta_iter, tol=tol)
        degenerate_x = xminus
        theta = theta_iter
        approx_saddle_position = 0 * degenerate_x
        for i in range(np.size(degenerate_x)):
            if np.abs(degenerate_x[i] - old_xminus[i]) > np.abs(degenerate_x[i] - old_xplus[i]):
                approx_saddle_position[i] = old_xminus[i]
            else:
                approx_saddle_position[i] = old_xplus[i]
        return approx_saddle_position, theta

    if np.size(theta_comb) == 2:
        low_theta = min(theta_comb)
        high_theta = max(theta_comb)
        theta_comb = np.linspace(high_theta, low_theta, 50)

    hill, par, gamma = NDMAparsTHETA_to_boxyboxpars(theta_comb[0], par_NDMA, index_to_varry)
    old_theta = theta_comb[0]
    success, old_xminus, old_xplus, remainder = boxy_box_from_pars(hill, par, gamma, maxiter=300)
    approx_saddle_position, approx_saddle_theta = [], []
    for theta_iter in theta_comb[1:]:
        hill, par_iter, gamma = NDMAparsTHETA_to_boxyboxpars(theta_iter, par_NDMA, index_to_varry)
        success, xminus, xplus, remainder = boxy_box_from_pars(hill, par_iter, gamma, maxiter=300)
        if not success:
            continue
        if is_degenerate(xminus, xplus, tol=10**-5) != is_degenerate(old_xminus, old_xplus, tol=10**-5):
            coord, theta_val = outlier(xminus, xplus, theta_iter, old_xminus, old_xplus, old_theta, tol=10**-5)
            approx_saddle_position.append(coord)
            approx_saddle_theta.append(theta_val)
        old_xminus, old_xplus, old_theta = xminus, xplus, theta_iter

    return approx_saddle_position, approx_saddle_theta


def saddle_node_with_boxybox_THETA(saddle_node_problem, theta_comb, par43, index_to_varry):
    """ for the EMT saddle node problem taken a selection of hill coefficients and a parameter, it finds all saddle nodes"""
    if np.size(theta_comb) == 2:
        low_theta = min(theta_comb)
        high_theta = max(theta_comb)
        theta_comb = np.linspace(high_theta, low_theta, 50)
    approx_saddle_position, saddle_theta = approx_saddle_node_with_boxy_box_THETA(theta_comb, par43, index_to_varry)
    par_of_SNbif, bad_candidate = [], []
    for i in range(len(saddle_theta)):
        par43[index_to_varry] = saddle_theta[i]
        saddle = saddle_node_problem.find_saddle_node(index_to_varry, par43, equilibria=approx_saddle_position[i])
        if saddle:
            par_of_SNbif.append(saddle)
        else:
            bad_candidate.append(saddle_theta[i])
    return par_of_SNbif, bad_candidate


def eqs_with_boxyboxEMT(hillpar, par_NDMA):
    """
    takes some info on the
    """
    n, par, gamma = NDMApars_to_boxyboxpars(hillpar, par_NDMA)
    success, xminus, xplus, remainder = boxy_box_from_pars(n, par, gamma)
    all_corners = corners_of_box(xminus, xplus)
    NDMA_eqs = f.local_equilibrium_search(all_corners, hillpar, par_NDMA)
    NDMA_eqs = f.remove_doubles(NDMA_eqs, hillpar, par_NDMA, uniqueRootDigits=5)
    return NDMA_eqs


