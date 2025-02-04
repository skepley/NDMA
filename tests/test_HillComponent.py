"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 4/3/2020 
"""
import numpy as np
from ndma.activation.hill import HillActivation

def test_hill_component():

    # set some parameters to test
    ell = 1
    theta = 3
    delta = 5
    n = 4.1
    x0 = np.array([3, 4, 5, 6])
    # x0 = 3

    # test Hill component code
    H1 = HillActivation(-1, ell=ell, theta=theta, delta=delta,
                        hillCoefficient=n)  # A function of x with all parameters fixed
    H2 = HillActivation(-1, ell=ell, theta=theta)  # A function of x with callable variable parameters: {delta, n}
    pp0 = np.array([delta, n])
    H3 = HillActivation(-1)  # A function of x with all parameters callable
    parameterDict = {'ell': ell, 'theta': theta}
    H4 = HillActivation(-1, **parameterDict)  # Test construction by dictionary

    # check function calls
    assert np.all(H1(x0) == H2(x0, [delta, n]))
    assert np.all(H1(x0) == H3(x0, [ell, delta, theta, n]))

    # check derivative function calls
    assert np.all(H1.dx(x0, []) == H2.dx(x0, [delta, n]))
    assert np.all(H1.dx(x0, []) == H3.dx(x0, [ell, delta, theta, n]))

    assert np.all(H1.dx2(x0, []) == H2.dx2(x0, [delta, n]))
    assert np.all(H1.dx2(x0, []) == H3.dx2(x0, [ell, delta, theta, n]))

    # print(H1.dn(x0)) # HANDLE THIS EXCEPTION
    assert np.all(H2.diff(x0, [delta, n], 0)== H3.diff(x0, [ell, delta, theta, n], 1))

    # print(H1.dndx(x0))
    assert np.all(H2.dxdiff(x0, [delta, n], 0) == H3.dxdiff(x0, [ell, delta, theta, n], 1))
