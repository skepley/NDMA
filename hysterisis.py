"""
Setting up the hysteresis problem, for the Toggle Switch case

    Other files required: hill_model, saddle_node, models

    Author: Elena Queirolo
"""

from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch
from scipy.optimize import minimize


def wrapper_minimization(HM, starting_pars, parameterIndex=1):
    # this function takes as input a Hill Model, initial parameter values and the index of the parameter and maximises
    # the hysteresis parameters

    SN = SaddleNode(HM)

    # find starting point
    starting_values = SN.find_saddle_node(starting_pars, parameterIndex)

    # restructuring starting values in the right format
    # the right format: [lambda1, x1, v1, lambda2, x2, v2, other_pars]

    # create contraints
    constraint_hysteresis = default_constraints(SN, parameterIndex)

    # create minimizing function
    min_function = negative_distance(parameterIndex)

    results = minimize(min_function, starting_values, method='SLSQP', jac='True', constraints=constraint_hysteresis)
    return results


def negative_distance(parameterIndex):
    def compute_distance(variables):
        gamma0 = variables[parameterIndex]
        gamma1 = variables[parameterIndex + 5]
        fun = gamma0 - gamma1
        dim_jac = np.shape(variables)[0]
        jac = np.zeros([dim_jac, dim_jac])
        jac[parameterIndex] = 1
        jac[parameterIndex + 5] = -1
        return fun, jac
    return compute_distance


def one_saddlenode_problem(SN, first_or_second, paramIndex):
    def saddle_node_problem(variables):
        fixed_pars = variables[-8:]
        u_and_v_index0 = 1 + (first_or_second - 1)*5
        u_and_v = variables[u_and_v_index0:u_and_v_index0+5]
        gamma = variables[u_and_v_index0-1]
        all_vars = [u_and_v, fixed_pars[0:paramIndex], gamma, fixed_pars[paramIndex:]]
        # warning: not sure if diff does the trick here
        return SN(all_vars), SN.diff(all_vars)
    return saddle_node_problem


def default_constraints(SN, paramIndex=1):

    def wrap_in_constraint(first_or_second):
        function_loc, function_jac = one_saddlenode_problem(SN, first_or_second, paramIndex)
        dic = { "type": "eq",
                "fun": function_loc,
                "jac": function_jac}
        return dic

    list_of_dics = [wrap_in_constraint(1), wrap_in_constraint(2)]
    return list_of_dics


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
SN_main = SaddleNode(f)

# ==== find saddle node for a parameter choice
rho = 4.1
p = np.array([1, 1, 5, 3, 1, 1, 6, 3.5], dtype=float)


def distance_func(p_loc, SN_loc=SN_main, rho_loc=rho):
    dist = hysteresis(p_loc, SN_loc, rho_loc)
    return -dist


distance = distance_func(p)
print('Distance = ', distance)
res = minimize(distance_func, p, method='Nelder-Mead')

print('Minimal distance = ', res)


