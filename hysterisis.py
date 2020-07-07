# import numpy as np
# import scipy
# from hill_model import *
# from itertools import product
# from saddle_node import *
# from models import ToggleSwitch
#
#
# def find_max_hysteresis(SNode, parameters, gamma0=1, x0=[1, 2]):
#     # core element
#     def distance_function(param):
#         return hysteresis(SNode, param, gamma0, x0)
#
#     optimum_parameters = scipy.optimize.minimize(distance_function, parameters, method='nelder-mead')
#     return optimum_parameters
#
#
# def hysteresis(SN, par0, gamma0, x0):
#     # find two different saddles and return their distance in "gamma"
#     jSearchNodes = np.linspace(gamma0/10, 10 * gamma0, 25)
#     # par0 vector of all parameters == check what par0 !
#     parameters = ezcat(par0)
#     jSols = SN.find_saddle_node(0, parameters[0], par0,  freeParameterValues=jSearchNodes)
#     gamma1 = jSols[0]
#     gamma2 = jSols[-1]
#     return np.abs(gamma1-gamma2)
#
#
# # testing
# decay = np.array([1, 1], dtype=float)  # gamma
# p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
# p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
# f = ToggleSwitch(decay, [p1, p2])
# n0 = 4.1
# SN_main = SaddleNode(f)
#
# # ==== find saddle node minimizer for some initial parameter choice
# p0 = np.array([1, 5, 3, 1, 6, 3],
#               dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
# p1 = np.array([1, 4, 3, 1, 1, 5, 3], dtype=float)
#
# localMinimum = find_max_hysteresis(SN_main, p1)
# print(localMinimum)


"""
Testing and analysis for SaddleNode and ToggleSwitch classes

    Other files required: hill_model, saddle_node, models

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/24/20; Last revision: 6/24/20
"""
import numpy as np
import matplotlib.pyplot as plt
from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch
from scipy.optimize import minimize


def hysteresis(SN_loc, rho_loc, p_loc):
    fullParameter = ezcat(rho_loc, p_loc)
    j = 1
    jSearchNodes = np.linspace(fullParameter[j] / 10, 10 * fullParameter[j], 25)
    print(jSearchNodes)
    jSols = SN_loc.find_saddle_node(j, rho_loc, p_loc, freeParameterValues=jSearchNodes)
    if len(jSols) is 2:
        distance_loc = abs(jSols[1] - jSols[0])
    else:
        return 0
    for sol in jSols:
        pSol = fullParameter.copy()
        pSol[j] = sol
        f.plot_nullcline(pSol)
    plt.title('parameter: {0}'.format(j))
    return distance_loc


np.set_printoptions(precision=2, floatmode='maxprec_equal')

# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.components[0]
H2 = f2.components[0]

p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
SN = SaddleNode(f)

# ==== find saddle node for a parameter choice
rho = 4.1
p = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)


# x0Sol, v0Sol, rhoSol = [u0Sol.x[idx] for idx in [[0, 1], [2, 3], [4]]]
# # compare to rhoSol = [ 4.55637172,  2.25827744,  0.82199933, -0.56948846,  3.17447061]

# plot nullclines and equilibria
#plt.close('all')
#plt.figure()
#f.plot_nullcline(rho, p)
#plt.title('Initial parameters: \n' + np.array2string(ezcat(rho, p)))


def distance_func(p_loc):
    dist = hysteresis(SN, rho, p_loc)
    return -dist


distance = distance_func(p)
print('Distance = ', distance)
res = minimize(distance_func, p, method='nelder-mead')




def snapshot_data(hillModel, N, parameter):
    """Get nullcline and equilibria data at a value of N and parameter"""

    equi = hillModel.find_equilibria(10, N, parameter)
    Z = np.zeros_like(Xp)

    # unpack decay parameters separately
    gamma = np.array(list(map(lambda f_i, parm: f_i.curry_gamma(parm)[0], hillModel.coordinates,
                              hillModel.unpack_variable_parameters(hillModel.parse_parameter(N, parameter)))))
    null1 = (hillModel(np.row_stack([Z, Yp]), N, parameter) / gamma[0])[0, :]  # f1 = 0 nullcline
    null2 = (hillModel(np.row_stack([Xp, Z]), N, parameter) / gamma[1])[1, :]  # f2 = 0 nullcline

    return null1, null2, equi


