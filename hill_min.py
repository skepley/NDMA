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
from scipy.optimize import BFGS


def wrapper_minimization(HM, starting_pars, parameterIndex=0, problem='hill_min', list_of_constraints=None):
    # this function takes as input a Hill Model, initial parameter values and the index of the parameter and maximises
    # the hysteresis parameters
    # INPUTS
    # HM        Hill model
    # starting_pars     starting parameters for minimization
    # parameterIndex=1  which parameter is the spacial one in the problem
    # problem='hysteresis'  what problem is to solve
    # list_of_constraints   are there any special constraints? Default = 'hysteresis' and 'norm_param'

    if list_of_constraints is None:
        list_of_constraints = ['saddle_node', 'norm_param']
    SN = SaddleNode(HM)

    # find starting point
    jSearchNodes = np.linspace(starting_pars[parameterIndex] / 10, 10 * starting_pars[parameterIndex], 25)
    starting_values = SN.find_saddle_node(parameterIndex, starting_pars, freeParameterValues=jSearchNodes,
                                          flag_return=1)

    # restructuring starting values in the right format
    # the right format: [lambda1, x1, v1, lambda2, x2, v2, other_pars]
    if starting_values.shape[0] > 1:
        raise Exception("We found more than 1 saddle node: " + str(starting_values.shape[0]))
    elif starting_values.shape[0] < 1:
        raise Exception("We couldn't find 1 saddle node")
    non_special_starting_pars = ezcat(starting_pars[:parameterIndex], starting_pars[parameterIndex + 1:])
    all_starting_values = ezcat(starting_values[0, :], starting_values[1, :], non_special_starting_pars)

    # create constraints
    all_constraints = list()
    if 'saddle_node' in list_of_constraints + [problem]:
        all_constraints = all_constraints + saddle_node_constraint(SN, parameterIndex)
    # if 'norm_param' in list_of_constraints:
    #    all_constraints = all_constraints + parameter_norm_constraint(all_starting_values)

    # create minimizing function
    if problem is 'saddle_node':
        min_function, jac_func, hessian_func = hill_coefficient_min()
    else:
        raise Exception("Not coded yet = only hysteresis considered")
    print(all_starting_values)
    results_min = minimize(min_function, all_starting_values, method='trust-constr', jac=jac_func,
                           constraints=all_constraints, hess=hessian_func)
    return results_min


def hill_coefficient_min(SN_loc):
    def compute_hill(variables):
        n = SN_loc.model.dimension
        hill = variables[2*n]
        return hill

    def jacobian_hill(variables):
        n = SN_loc.model.dimension
        dim_par = np.shape(variables)[0]
        jac = np.zeros([dim_par])
        jac[2*n] = 1
        return jac

    def hessian_hill(variables):
        dim_par = np.shape(variables)[0]
        hess = np.zeros([dim_par, dim_par])
        return hess

    return compute_hill, jacobian_hill, hessian_hill


def saddle_node_constraint(SN_loc):
    def saddle_node_problem(variables):
        return SN_loc(variables)

    def saddle_node_jac(variables):
        saddle_diff = SN_loc.diff(variables)
        return saddle_diff

    def saddle_node_hess(variables, vector):
        hessian_full = SN.diff2(variables)
        hessian_prod = np.einsum('ijk,i', hessian_full, vector)
        return hessian_prod

    return saddle_node_problem, saddle_node_jac, saddle_node_hess

def parameter_norm(initial_sol):
    def norm_par(parameter):
        norm_los = np.sum(np.abs(parameter)) - np.sum(np.abs(initial_sol))
        print("norm = ", norm_los)
        return norm_los

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


def parameter_norm_constraint(initial_sol):
    function_loc, function_jac, func_hessian = parameter_norm(initial_sol)
    constraint = scipy.optimize.NonlinearConstraint(fun=function_loc, lb=0, ub=0, jac=function_jac, hess=func_hessian)
    list_constr = [constraint]
    return list_constr


np.set_printoptions(precision=5, floatmode='maxprec_equal')

# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])

SN = SaddleNode(f)

# ==== find saddle node for a parameter choice
hill = 4.1
p = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)


long_p = np.append([hill], p)
results = wrapper_minimization(f, long_p)
print('Success: ', results.success)

exit()
# variables_example = ezcat(gamma_0, x0, v0, gamma_1, x1, v1, hill, [other_pars])
