import numpy as np

from ndma.boxy_box import boxy_box
from ndma.model import Model

def point_is_in_box(point, low_box, high_box):
    for i in range(np.shape(low_box)[0]):
        if not low_box[i]-10**-2<point[i]<high_box[i]+10**-2:
            print(point[i], low_box[i], high_box[i])
            return False
    return True

def test_boxybox():
    TS_model = Model.Model_from_string('x: (~y)\n y: (~x)')
    gamma = 1.
    ell, delta, theta, hill = 1, 1.9, 1.5, 5
    parameters_bistable = np.array([gamma, ell, delta, theta, hill, gamma, ell, delta, theta, hill])
    parameter_monostable = np.array([gamma, ell, 0.1, theta, hill, gamma, ell, 0.2, theta, hill])
    x = np.array([1, 2.1])
    y = TS_model(x, parameters_bistable)

    x_low, x_high = boxy_box(TS_model, parameters_bistable)

    x_0 = np.array([5, 6.7])  # "big" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameters_bistable).y[:,-1]
    assert point_is_in_box(y_0, x_low, x_high)

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameters_bistable).y[:,-1]
    assert point_is_in_box(y_0, x_low, x_high)

    x_low, x_high = boxy_box(TS_model, parameter_monostable)

    x_0 = np.array([5, 6.7])  # "big" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameter_monostable).y[:,-1]
    assert point_is_in_box(y_0, x_low, x_high)

    x_0 = np.array([0.1, 0.001])  # "small" initial point
    T = [0, 300.0]
    y_0 = TS_model.odeint(T, x_0, parameter_monostable).y[:,-1]
    assert point_is_in_box(y_0, x_low, x_high)