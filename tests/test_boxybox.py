import numpy as np

from ndma.boxy_box import boxy_box
from ndma.model import Model

def point_is_in_box(point, low_box, high_box):
    for i in range(np.shape(low_box)[0]):
        if not low_box[i]-10**-2<point[i]<high_box[i]+10**-2:
            print(point[i], low_box[i], high_box[i])
            return False
    return True


def boxy_box_call(model, parameter, x_0, T):
    x_low, x_high = boxy_box(model, parameter)
    y_0 = model.odeint(T, x_0, parameter).y[:, -1]
    return point_is_in_box(y_0, x_low, x_high)

def test_boxybox_TS():
    TS_model = Model.Model_from_string('x: (~y)\n y: (~x)')
    gamma = 1.
    ell, delta, theta, hill = 1, 1.9, 1.5, 5
    parameters_bistable = np.array([gamma, ell, delta, theta, hill, gamma, ell, delta, theta, hill])
    parameter_monostable = np.array([gamma, ell, 0.1, theta, hill, gamma, ell, 0.2, theta, hill])
    x = np.array([1, 2.1])
    y = TS_model(x, parameters_bistable)

    T = [0, 300.0]

    x_0 = np.array([5, 6.7]) # "big" initial point
    assert boxy_box_call(TS_model, parameters_bistable, x_0, T)

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    assert boxy_box_call(TS_model, parameters_bistable, x_0, T)

    x_0 = np.array([5, 6.7])  # "big" initial point
    assert boxy_box_call(TS_model, parameter_monostable, x_0, T)

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    assert boxy_box_call(TS_model, parameter_monostable, x_0, T)


def test_boxybox_othermodel():
    other_model = Model.Model_from_string('x: x(~y)\n y: (~x)')
    gamma = 1.
    ell, delta, theta, hill = 1, 1.9, 1.5, 5
    a_parameter = np.array([gamma, ell, delta, theta, hill, ell, delta, theta, hill, gamma, ell, delta, theta, hill])
    another_parameter = np.array([gamma, ell, 0.1, theta, hill, ell, delta, theta, hill, gamma, ell, 0.2, theta, hill])
    x = np.array([1, 2.1])
    y = other_model(x, a_parameter)

    T = [0, 300.0]

    x_0 = np.array([5, 6.7])  # "big" initial point
    assert boxy_box_call(other_model, a_parameter, x_0, T)

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    assert boxy_box_call(other_model, a_parameter, x_0, T)

    x_0 = np.array([5, 6.7])  # "big" initial point
    assert boxy_box_call(other_model, another_parameter, x_0, T)

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    assert boxy_box_call(other_model, another_parameter, x_0, T)


def test_boxybox_3Dmodel():
    other_3Dmodel = Model.Model_from_string('x: (~z)\n y: (~x)\n z: (~y)')
    gamma = 1.
    ell, delta, theta, hill = 1, 1.9, 1.5, 5
    a_parameter = np.array([gamma, ell, delta, theta, hill, gamma, ell, delta, theta, hill, gamma, ell, delta, theta, hill])
    another_parameter = np.array([gamma, ell, 0.1, theta, hill, gamma+0.1, ell, delta, theta, hill, gamma, ell, 0.2, theta, hill])
    x = np.array([1, 2.1, 4])
    y = other_3Dmodel(x, a_parameter)

    T = [0, 300.0]

    x_0 = np.array([5, 6.7, 8])  # "big" initial point
    assert boxy_box_call(other_3Dmodel, a_parameter, x_0, T)

    x_0 = np.array([0.1, 0.001, 0.12])  # "small" initial point
    assert boxy_box_call(other_3Dmodel, a_parameter, x_0, T)

    x_0 = np.array([5, 6.7, 11])  # "big" initial point
    assert boxy_box_call(other_3Dmodel, another_parameter, x_0, T)

    x_0 = np.array([0.1, 0.001, 0.2])  # "small" initial point
    assert boxy_box_call(other_3Dmodel, another_parameter, x_0, T)