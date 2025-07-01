"""
The boxy box algorithm implemented in all generality

The boxy box algorithm is an iterative algorithm to bound all equilibria of a Model with a monotone factorisation.
A model with monotone factorisation has the property that is can be written as

dot x = - gamma x + sigma_minus (x) sigma_plus(x)

where sigma_minus is a monotonically decreasing bounded function, and sigma_plus is a monotonically increasing
bounded function.
Thus, if all equilibria are in the interval [x_min, x_max], they are in the interval
[sigma_minus(x_max)sigma_plus(x_min)/gamma, sigma_minus(x_min)sigma_plus(x_max)/gamma]
this being the iterative step.
The algorithm is initialised as the mathematical bounds of each coordinate - based on the "image" functionality in the
Activation Functions
"""
import itertools
import warnings

import numpy as np
from ndma.model import Model, RestrictedHillModel

"""
Remark: maybe someone better than me could code it better - but this is a working implementation that respects
the math and did not require too much time to set up - Elena Queirolo, 9.02.2025

In this implementation, a little hack is used: instead of constructing the functions sigma_plus and sigma_minus from
scratch, we realise they are Models where gamma = 0 and only the correct non-linearities have been picked.
Thus, the first step of this algorithm will be to properly construct the wanted Models. Then, the iterative step will
follow trivially
"""

# # # MAIN BOXY BOX ALGORITHM
def has_monotone_factorisation(model):
    for coordinate in model.coordinates:
        signs = [component.sign for component in coordinate.productionComponents]
        type = coordinate.productionType
        # type groups the summands, signs tell us the monotonicity of each
        # thus, we check inside each summand that all have the same monotonicity
        indices = np.cumsum(np.array(type))
        start_index = 0
        for j in indices:
            if any([sign != signs[start_index] for sign in signs[start_index:j]]):
                return False
            start_index = j
    return True

def create_sigmas(model, param):
    def flatten(xss):
        return [x for xs in xss for x in xs]
    activation_function = model.coordinates[0].activation
    n_param_for_activation = len(activation_function.PARAMETER_NAMES)
    unpacked_param = model.unpack_parameter(param)
    param_without_gamma = [param[1:] for param in unpacked_param]
    new_gamma = 0 * np.ones(model.dimension)

    # we need to compute the new
    # productionSign, productionType, productionIndex
    productionIndex = model.productionIndex
    new_sign_minus, new_sign_plus, new_type_minus, new_type_plus, new_production_minus, new_production_plus = [], [], [], [], [], []
    param_minus, param_plus = [], []
    starting_box = np.empty([model.dimension, 2])
    for i in range(model.dimension):
        coordinate = model.coordinates[i]
        starting_box[i, :] = coordinate.eq_interval(unpacked_param[i])
        signs = [component.sign for component in coordinate.productionComponents]
        type = coordinate.productionType
        index = productionIndex[i]

        new_sign_minus.append([])
        new_sign_plus.append([])
        new_type_minus.append([])
        new_type_plus.append([])
        new_production_minus.append([])
        new_production_plus.append([])
        param_minus.append([])
        param_plus.append([])

        low_index = 0
        for high_index in type:
            param_loc = param_without_gamma[i][low_index*n_param_for_activation:(low_index + high_index)*n_param_for_activation]
            param_loc = np.reshape(param_loc, (high_index, n_param_for_activation))
            if signs[low_index] == 1:
                [new_sign_plus[i].append(sign) for sign in signs[low_index:low_index+high_index]]
                new_type_plus[i].append(high_index)

                for k in index[low_index:low_index+high_index]:
                    new_production_plus[i].append(k)
                param_plus[i].append(param_loc)
            else:
                [new_sign_minus[i].append(sign) for sign in signs[low_index:low_index+high_index]]
                new_type_minus[i].append(high_index)
                for k in index[low_index:low_index+high_index]:
                    new_production_minus[i].append(k)
                param_minus[i].append(param_loc)
            low_index = high_index + low_index
    if len(flatten(new_sign_minus)) < 1:
        new_model_minus = lambda x: 0*x + 1
    else:
        new_model_minus = Model(new_gamma, param_minus, new_sign_minus, new_type_minus, new_production_minus, activation_function)
    if len(flatten(new_sign_plus)) < 1:
        new_model_plus = lambda x: 0 * x + 1
    else:
        new_model_plus = Model(new_gamma, param_plus, new_sign_plus, new_type_plus, new_production_plus, activation_function)
    x_min, x_max = starting_box[:,0], starting_box[:,1]
    return new_model_minus, new_model_plus, x_min, x_max

def extract_gamma(model, param):
    unpacked_param = model.unpack_parameter(param)
    gamma = np.array([param[0] for param in unpacked_param])
    return gamma

def boxy_box(model: Model, *parameters, maxiter=100):
    if not has_monotone_factorisation(model):
        ValueError("The boxy box algorithm cannot be applied if the model doesn't have a monotone factorisation.")
    if isinstance(model, RestrictedHillModel):
        ValueError("This implementation requests Model inputs instead of RestrictedHillModel.")
    tol = 10**-6
    sigma_minus, sigma_plus, x_min, x_max = create_sigmas(model, *parameters)
    gamma = extract_gamma(model, *parameters)
    old_x_min, old_x_max = x_min, x_max
    for i in range(maxiter):
        x_min = sigma_minus(x_max)*sigma_plus(x_min)/gamma
        x_max = sigma_minus(x_min)*sigma_plus(x_max)/gamma
        if np.linalg.norm(old_x_min - x_min) < tol and np.linalg.norm(old_x_max - x_max) < tol:
            break
        old_x_min, old_x_max = x_min, x_max
        if i == 99:
            warnings.warn("The boxy box algorithm did not converge")
    return x_min, x_max

## DETECTING EQUILIBRIA WITH BOXY BOX
def equilibria_with_boxybox(model, parameters):
    """
    takes some info on the
    """
    xminus, xplus = boxy_box(model, parameters)
    all_corners = corners_of_box(xminus, xplus)
    doubled_eqs = model.local_equilibrium_search(all_corners, parameters)
    equilibria = model.remove_doubles(doubled_eqs, parameters, uniqueRootDigits=5)
    return equilibria


## SADDLE DETECTION WITH BOXY BOX
def approx_saddle_node_with_boxy_box(model, par_comb, all_pars):
    """ by going through all coefficients stored in par_comb, we look for changes in the number of equilibria and
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

    if np.size(par_comb) == 2:
        low_par = min(par_comb)
        high_par = max(par_comb)
        par_comb = np.linspace(high_par, low_par, 50)
    old_par = par_comb[0]
    old_xminus, old_xplus = boxy_box(model, old_par, all_pars, maxiter=300)
    approx_saddle_position, approx_saddle_hill = [], []
    for par_iter in par_comb[1:]:
        success, xminus, xplus, remainder = boxy_box(model, par_iter, all_pars, maxiter=300)
        if not success:
            continue
        if is_degenerate(xminus, xplus, tol=10**-5) != is_degenerate(old_xminus, old_xplus, tol=10**-5):
            coord, hill_val = outlier(xminus, xplus, par_iter, old_xminus, old_xplus, old_par, tol=10**-5)
            approx_saddle_position.append(coord)
            approx_saddle_hill.append(hill_val)
        old_xminus, old_xplus, old_par = xminus, xplus, par_iter

    return approx_saddle_position, approx_saddle_hill


def saddle_node_with_boxybox(model, saddle_node_problem, par_comb, all_par):
    """ for the EMT saddle node problem taken a selection of hill coefficients and a parameter, it finds all saddle nodes"""
    if np.size(par_comb) == 2:
        low_par = min(par_comb)
        high_par = max(par_comb)
        par_comb = np.linspace(high_par, low_par, 50)
    approx_saddle_position, saddle_par = approx_saddle_node_with_boxy_box(model,par_comb, all_par)
    par_of_SNbif, bad_candidate = [], []
    for i in range(len(saddle_par)):
        saddle = saddle_node_problem.find_saddle_node(0, saddle_par[i], all_par, equilibria=approx_saddle_position[i])
        if saddle:
            par_of_SNbif.append(saddle)
        else:
            bad_candidate.append(saddle_par[i])
    return par_of_SNbif, bad_candidate

## SMALL SUPPORT FUNCTIONS
def is_degenerate(xminus, xplus, tol=10**-7):
    if np.linalg.norm(xminus - xplus) < tol:
        return True
    return False

def corners_of_box(xminus, xplus):
    all_corners = list(itertools.product(*zip(xminus, xplus)))
    return np.array(all_corners)
