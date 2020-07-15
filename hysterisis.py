"""
Setting up the hysteresis problem, for the Toggle Switch case

    Other files required: hill_model, saddle_node, models

    Author: Elena Queirolo
"""

from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch
from scipy.optimize import minimize


def wrapper_minimization(HM, starting_pars, parameterIndex=1, problem='hysteresis', list_of_constraints=None):
    # this function takes as input a Hill Model, initial parameter values and the index of the parameter and maximises
    # the hysteresis parameters
    # INPUTS
    # HM        Hill model
    # starting_pars     starting parameters for minimization
    # parameterIndex=1  which parameter is the spacial one in the problem
    # problem='hysteresis'  what problem is to solve
    # list_of_constraints   are there any special constraints? Default = 'hysteresis' and 'norm_param'

    if list_of_constraints is None:
        list_of_constraints = ['hysteresis', 'norm_param']
    SN = SaddleNode(HM)

    # find starting point
    starting_values = SN.find_saddle_node(parameterIndex, starting_pars, flag_return=1)

    # restructuring starting values in the right format
    # the right format: [lambda1, x1, v1, lambda2, x2, v2, other_pars]
    if starting_values.shape[0] > 2:
        raise Exception("We found more than 2 saddle nodes " + str(starting_values.shape[0]))
    elif starting_values.shape[0] < 2:
        raise Exception("We couldn't find 2 saddle nodes, we only found " + str(starting_values.shape[0]))
    all_starting_values = ezcat(starting_values[0, :], starting_values[1, :], starting_pars)

    # create constraints
    all_constraints = list()
    if 'hysteresis' in list_of_constraints + [problem]:
        all_constraints = all_constraints + hysteresis_constraints(SN, parameterIndex)
    if 'norm_param' in list_of_constraints:
        all_constraints = all_constraints + parameter_norm_constraint()

    # create minimizing function
    if problem is 'hysteresis':
        min_function = negative_distance(parameterIndex)
    else:
        raise Exception("Not coded yet = only hysteresis considered")

    results = minimize(min_function, all_starting_values, method='SLSQP', jac='True', constraints=all_constraints[0])
    return results


def negative_distance(parameterIndex):
    def compute_distance(variables):
        gamma0 = variables[parameterIndex]
        gamma1 = variables[parameterIndex + 5]
        fun = gamma0 - gamma1
        dim_jac = np.shape(variables)[0]
        jac = np.zeros([dim_jac])
        jac[parameterIndex] = 1
        jac[parameterIndex + 5] = -1
        return fun, jac
    return compute_distance


def one_saddlenode_problem(SN_loc, first_or_second, paramIndex):
    def saddle_node_problem(variables):
        n = SN_loc.model.dimension
        num_pars = SN_loc.model.nVariableParameter
        fixed_pars = variables[-num_pars:]
        index_gamma = (first_or_second - 1)*(2 * n + 1)
        u_and_v_index0 = 1 + index_gamma
        u_and_v = variables[u_and_v_index0:u_and_v_index0 + 2*n]
        gamma = variables[index_gamma]
        all_vars = ezcat(u_and_v, fixed_pars[0:paramIndex], gamma, fixed_pars[paramIndex:])

        return SN_loc(all_vars)

    def saddle_node_jac(variables):
        # variables = lambda1, x1, v1, lambda2, x2, v2, other_pars
        n = SN_loc.model.dimension
        num_pars = SN_loc.model.nVariableParameter
        fixed_pars = variables[-num_pars:]

        index_gamma = (first_or_second - 1)*(2 * n + 1)
        u_and_v_index0 = 1 + index_gamma
        u_and_v = variables[u_and_v_index0:u_and_v_index0 + 2*n]
        gamma = variables[index_gamma]
        all_vars = ezcat(u_and_v, fixed_pars[0:paramIndex], gamma, fixed_pars[paramIndex:])
        non_zero_diff = SN_loc.diff(all_vars)

        # reordering of the result w.r.t. the global ordering
        full_derivative = np.zeros([non_zero_diff.shape[0], len(variables)])
        #
        full_derivative[:, index_gamma] = non_zero_diff[:, 2*n + paramIndex]
        full_derivative[:, index_gamma+1: index_gamma+2 * n + 1] = non_zero_diff[:, 0:2*n]

        indices_fixed_pars = np.arange(num_pars)
        indices_fixed_pars = np.delete(indices_fixed_pars, paramIndex)
        full_derivative[:, -num_pars+1:] = non_zero_diff[:, 2*n+indices_fixed_pars]
        return full_derivative
    return saddle_node_problem, saddle_node_jac


def hysteresis_constraints(SN, paramIndex=1):

    def wrap_in_constraint(first_or_second):
        function_loc, function_jac = one_saddlenode_problem(SN, first_or_second, paramIndex)
        dic = { "type": "eq",
                "fun": function_loc,
                "jac": function_jac}
        return dic

    list_of_dics =list()
    list_of_dics.append(wrap_in_constraint(1))
    list_of_dics.append(wrap_in_constraint(2))
    return list_of_dics


def parameter_norm():
    def norm_par(parameter):
        return np.sum(parameter)

    def jac_norm_par(parameters):
        jac = np.ones(len(parameters))
        return jac
    return norm_par, jac_norm_par


def parameter_norm_constraint():
    function_loc, function_jac = parameter_norm()
    dic = { "type": "eq",
            "fun": function_loc,
            "jac": function_jac}
    list_of_dic = [dic]
    return list_of_dic


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




def distance_func(p_loc, SN_loc=SN, rho_loc=rho):
    dist = hysteresis(p_loc, SN_loc, rho_loc)
    return -dist


distance = distance_func(p)
print('Distance = ', distance)
#res = minimize(distance_func, p, method='Nelder-Mead')
#
#print('Minimal distance = ', res)


long_p = np.append([rho], p)
results = wrapper_minimization(f, long_p)
print(results)
#print('Minimal distance = ', res)


