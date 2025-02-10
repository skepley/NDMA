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
The algorithm cannot be initialised by [x_min, x_max] = [-infty, infty], that would give a meaningful next step thanks
to the boundedness of the non-linearity.... but somehow returns nans
Instead a TEMPORARY hack is to take x_min = 0 and x_max = 100 TODO: fix the hack
"""
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
    for i in range(model.dimension):
        coordinate = model.coordinates[i]
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
            if signs[low_index] == 1:
                [new_sign_plus[i].append(sign) for sign in signs[low_index:low_index+high_index]]
                new_type_plus[i].append(high_index)
                new_production_plus[i].append(*index[low_index:low_index+high_index])
                [param_plus[i].append(param) for param in param_loc]
            else:
                [new_sign_minus[i].append(sign) for sign in signs[low_index:low_index+high_index]]
                new_type_minus[i].append(high_index)
                new_production_minus[i].append(*index[low_index:low_index+high_index])
                [param_minus[i].append(param) for param in param_loc]
            low_index = high_index + low_index
    if len(flatten(new_sign_minus)) < 1:
        new_model_minus = lambda x: 0*x + 1
    else:
        new_model_minus = Model(new_gamma, param_minus, new_sign_minus, new_type_minus, new_production_minus, activation_function)
    if len(flatten(new_sign_plus)) < 1:
        new_model_plus = lambda x: 0 * x + 1
    else:
        new_model_plus = Model(new_gamma, param_plus, new_sign_plus, new_type_plus, new_production_plus, activation_function)
    return new_model_minus, new_model_plus


def extract_gamma(model, param):
    unpacked_param = model.unpack_parameter(param)
    gamma = np.array([param[0] for param in unpacked_param])
    return gamma

def boxy_box(model: Model, *parameters):
    if not has_monotone_factorisation(model):
        ValueError("The boxy box algorithm cannot be applied if the model doesn't have a monotone factorisation.")
    if isinstance(model, RestrictedHillModel):
        ValueError("This implementation requests Model inputs instead of RestrictedHillModel.")
    sigma_minus, sigma_plus = create_sigmas(model, *parameters)
    gamma = extract_gamma(model, *parameters)
    x_min = 0 * np.ones(model.dimension)
    x_max = 100 * np.ones(model.dimension)
    for i in range(30):
        x_min = sigma_minus(x_max)*sigma_plus(x_min)/gamma
        x_max = sigma_minus(x_min)*sigma_plus(x_max)/gamma
    return x_min, x_max


if __name__ == "__main__":
    TS_model = Model.Model_from_string('x: (~y)\n y: (~x)')
    gamma = 1.
    ell, delta, theta, hill = 1, 1.9, 1.5, 5
    parameters_bistable = np.array([gamma, ell, delta, theta, hill, gamma, ell, delta, theta, hill])
    parameter_monostable = np.array([gamma, ell, 0.1, theta, hill, gamma, ell, 0.2, theta, hill])
    x = np.array([1,2.1])
    y = TS_model(x, parameters_bistable)
    print(TS_model,'\ncomputed at ',x, ' = ', y)

    print('\n\nBISTABLE boxy box on the model\n')
    x_low, x_high = boxy_box(TS_model, parameters_bistable)
    print('the boxy box returns ', x_low, x_high)

    print('Compare the final result with the forward integration of a point')

    x_0 = np.array([5,6.7]) # "big" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameters_bistable)
    print('final point = ', y_0.y[:,-1])

    x_0 = np.array([0.1,0.001]) # "small" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameters_bistable)
    print('final point = ', y_0.y[:,-1])

    print('\n\nMONOSTABLE boxy box\n')
    x_low, x_high = boxy_box(TS_model, parameter_monostable)
    print('the boxy box returns ', x_low, x_high)

    print('Compare the final result with the forward integration of a point')

    x_0 = np.array([5, 6.7])  # "big" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameter_monostable)
    print('final point = ', y_0.y[:, -1])

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameter_monostable)
    print('final point = ', y_0.y[:, -1])
