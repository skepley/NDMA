import numpy as np
import matplotlib.pyplot as plt
import itertools
from models.EMT_model import *


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
    gplus = lambda x: np.array((1,
                                1,
                                Hplus(n, par[4, :], x[0]),
                                1,
                                Hplus(n, par[8, :], x[2]),
                                1))

    gminus = lambda x: np.array((Hmin(n, par[0, :], x[1]) * Hmin(n, par[1, :], x[3]),
                                 Hmin(n, par[2, :], x[2]) * Hmin(n, par[3, :], x[4]),
                                 Hmin(n, par[5, :], x[5]),
                                 Hmin(n, par[6, :], x[4]),
                                 Hmin(n, par[7, :], x[1]) * Hmin(n, par[9, :], x[3]),
                                 Hmin(n, par[10, :], x[2]) * Hmin(n, par[11, :], x[4])))
    return gplus, gminus


def F_func(n, par, gamma):
    gplus, gminus = gpm_func(n, par)
    F = lambda x: - gamma * x + gplus(x) * gminus(x)
    return F


def phi_func(n, par, gamma):
    gplus, gminus = gpm_func(n, par)
    phi = lambda xplus, xminus: (gplus(xplus) * gminus(xminus) / gamma, gplus(xminus) * gminus(xplus) / gamma)
    return phi


def convergence(F, xminus, xplus):
    tol = 10 ** -7
    if np.linalg.norm(xminus - xplus) < tol:
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


def which_corner(F, xminus, xplus):
    tol = 10 ** -7
    allx = list(itertools.product(*zip(xminus, xplus)))

    for i in range(2 ** 6):
        if (np.linalg.norm(F(allx[i]))) < tol:
            print(np.linalg.norm(F(allx[i])))


def boxy_box_from_pars(n, par, gamma, maxiter=180):
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
    while (not convergence(F, xminus, xplus)) and (iter < maxiter):
        xplus_new, xminus_new = phi(xplus, xminus)
        remainder = np.append(remainder, np.linalg.norm(xplus - xplus_new) + np.linalg.norm(xminus_new - xminus))
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


def corners_of_box(xminus, xplus):
    all_corners = list(itertools.product(*zip(xminus, xplus)))
    return np.array(all_corners)



if __name__ == "__main__":
    failed_iters = 0
    bistability = 0
    n = 15.

    not_coplanar = 0

    for j in range(10):
        par = np.random.random(size=(12, 3))
        gamma = np.random.random(size=(6))

        success, xminus, xplus, remainder = boxy_box_from_pars(n, par, gamma)

        if not success:
            failed_iters += 1
            # plt.semilogy(remainder[0:40])
        else:

            if np.linalg.norm(xplus - xminus) > 0.1:
                bistability += 1
    # plt.show()

    print('No corner equilibria ', failed_iters, 'times out of', j + 1)
    print('Bistability found ', bistability, 'times out of', j + 1)
    print('not_coplanar found ', not_coplanar, 'times out of', j + 1)

    # set EMT-specific elements
    gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
    edgeCounts = [2, 2, 2, 1, 3, 2]
    parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
    # production parameters as variable
    f = EMT(gammaVar, parameterVar)

    par_NDMA = np.abs(np.random.random(42))
    hill = 3.2
    n, par, gamma = NDMApars_to_boxyboxpars(hill, par_NDMA)
    print(hill, par_NDMA)
    print(n, par, gamma)
    success, xminus, xplus, remainder = boxy_box_from_pars(n, par, gamma)
    all_corners = corners_of_box(xminus, xplus)
    norms = [np.linalg.norm(f(all_corners[i, :], hill, par_NDMA)) for i in range(np.size(all_corners, 1))]
    print('norms = ', norms)

    F_box = F_func(n, par, gamma)
    print('F_box =' ,F_box(all_corners[0, :]), '\nF_ndma =', f(all_corners[0, :], hill, par_NDMA))

