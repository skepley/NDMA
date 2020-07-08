"""
Setting up the hysteresis problem, for the Toggle Switch case

    Other files required: hill_model, saddle_node, models

    Author: Elena Queirolo
"""

from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch
from scipy.optimize import minimize


def hysteresis(p_loc, SN_loc, rho_loc):
    print('p_loc = ', p_loc)
    fullParameter = ezcat(rho_loc, p_loc)
    j = 1
    jSearchNodes = np.linspace(fullParameter[j] / 10, 10 * fullParameter[j], 25)
    jSols = SN_loc.find_saddle_node(j, rho_loc, p_loc, freeParameterValues=jSearchNodes)
    if len(jSols) is 2:
        distance_loc = abs(jSols[1] - jSols[0])
    else:
        return 0
    print(distance_loc)
    return distance_loc


np.set_printoptions(precision=5, floatmode='maxprec_equal')

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
p = np.array([1, 1, 5, 3, 1, 1, 6, 3.5], dtype=float)


# x0Sol, v0Sol, rhoSol = [u0Sol.x[idx] for idx in [[0, 1], [2, 3], [4]]]
# # compare to rhoSol = [ 4.55637172,  2.25827744,  0.82199933, -0.56948846,  3.17447061]

# plot nullclines and equilibria
#plt.close('all')
#plt.figure()
#f.plot_nullcline(rho, p)
#plt.title('Initial parameters: \n' + np.array2string(ezcat(rho, p)))


def distance_func(p_loc, SN_loc=SN, rho_loc=rho):
    dist = hysteresis(p_loc, SN_loc, rho_loc)
    return -dist


distance = distance_func(p)
print('Distance = ', distance)
res = minimize(distance_func, p, method='Nelder-Mead')

print('Minimal distance = ', res)


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


