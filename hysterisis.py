"""
Setting up the hysteresis problem, for the Toggle Switch case

    Other files required: hill_model, saddle_node, models

    Author: Elena Queirolo
"""

from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch
from scipy.optimize import minimize
import scipy


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
        min_function, jac_func, hessian_func = negative_distance(parameterIndex)
    else:
        raise Exception("Not coded yet = only hysteresis considered")

    results_min = minimize(min_function, all_starting_values, method='trust-constr', jac=jac_func, constraints=all_constraints, hess=hessian_func)
    # needs to add hessian to all constraints and change constraints to NonLinearConstraint
    return results_min


def negative_distance(parameterIndex):
    def compute_distance(variables):
        gamma0 = variables[parameterIndex]
        gamma1 = variables[parameterIndex + 5]
        fun = gamma0 - gamma1
        return fun

    def jacobian_distance(variables):
        dim_par = np.shape(variables)[0]
        jac = np.zeros([dim_par])
        jac[parameterIndex] = 1
        jac[parameterIndex + 5] = -1
        return jac

    def hessian_distance(variables):
        dim_par = np.shape(variables)[0]
        hess = np.zeros([dim_par, dim_par])
        return hess
    return compute_distance, jacobian_distance, hessian_distance


def one_saddlenode_problem(SN_loc, first_or_second, paramIndex):
    def get_small_variables(variables):
        n = SN_loc.model.dimension
        num_pars = SN_loc.model.nVariableParameter
        fixed_pars = variables[-num_pars:]

        index_gamma = (first_or_second - 1) * (2 * n + 1)
        u_and_v_index0 = 1 + index_gamma
        u_and_v = variables[u_and_v_index0:u_and_v_index0 + 2 * n]
        gamma = variables[index_gamma]
        all_vars = ezcat(u_and_v, fixed_pars[0:paramIndex], gamma, fixed_pars[paramIndex:])
        return all_vars, index_gamma

    def saddle_node_problem(variables):

        all_vars, index_gamma = get_small_variables(variables)
        sd_rhs = SN_loc(all_vars)
        return sd_rhs

    def saddle_node_jac(variables):
        # variables = lambda1, x1, v1, lambda2, x2, v2, other_pars
        n = SN_loc.model.dimension
        num_pars = SN_loc.model.nVariableParameter
        all_vars, index_gamma = get_small_variables(variables)
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

    def saddle_node_hess(variables, vector):
        n = SN_loc.model.dimension
        num_pars = SN_loc.model.nVariableParameter
        dim_var = np.shape(variables)[0]
        all_vars, index_gamma = get_small_variables(variables)
        small_vector = get_small_variables(vector)
        non_zero_hessian = SN.diff2(all_vars)
        non_zero_prod = np.einsum('ijk,i', non_zero_hessian, small_vector)

        # now we have to transport the result to the adequate shape
        full_hessian = np.zeros([dim_var, dim_var])
        full_hessian[:, index_gamma] = non_zero_prod[:, 2*n + paramIndex]
        full_hessian[:, index_gamma+1: index_gamma+2 * n + 1] = non_zero_prod[:, 0:2*n]

        indices_fixed_pars = np.arange(num_pars)
        indices_fixed_pars = np.delete(indices_fixed_pars, paramIndex)
        full_hessian[:, -num_pars+1:] = non_zero_prod[:, 2*n+indices_fixed_pars]
        return full_hessian
    return saddle_node_problem, saddle_node_jac, saddle_node_hess


def hysteresis_constraints(SN_loc, paramIndex=1):

    def wrap_in_constraint(first_or_second):
        function_loc, function_jac, func_hessian = one_saddlenode_problem(SN_loc, first_or_second, paramIndex)
        constraint = scipy.optimize.NonlinearConstraint(fun=function_loc, lb=0, ub=0, jac=function_jac, hess=func_hessian)
        return constraint

    list_of_constr =list()
    list_of_constr.append(wrap_in_constraint(1))
    list_of_constr.append(wrap_in_constraint(2))
    return list_of_constr


def parameter_norm():
    def norm_par(parameter):
        return np.sum(np.abs(parameter)) - 3

    def jac_norm_par(parameters):
        jac = np.ones(len(parameters))
        # adjusting for absolute values
        jac[np.where(parameters < 0)] = - jac[np.where(parameters < 0)]
        return jac
    
    def hess_norm_par(parameters, vector):
        dim_par = np.shape(parameters)[0]
        hess = np.zeros([dim_par, dim_par])
        return hess
    return norm_par, jac_norm_par, hess_norm_par


def parameter_norm_constraint():
    function_loc, function_jac, func_hessian = parameter_norm()
    constraint = scipy.optimize.NonlinearConstraint(fun=function_loc, lb=0, ub=0, jac=function_jac, hess=func_hessian)
    list_constr = [constraint]
    return list_constr


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


