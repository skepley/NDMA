"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 4/3/2020 
"""
import numpy as np
from hill_model import HillComponent

# set some parameters to test
ell = 1
theta = 3
delta = 5
n = 4.1
x0 = np.array([3, 4, 5, 6])
# x0 = 3

# test Hill component code
H1 = HillComponent(-1, ell=ell, theta=theta, delta=delta,
                   hillCoefficient=n)  # A function of x with all parameters fixed
H2 = HillComponent(-1, ell=ell, theta=theta)  # A function of x with callable variable parameters: {delta, n}
pp0 = np.array([delta, n])
H3 = HillComponent(-1)  # A function of x with all parameters callable
parameterDict = {'ell': ell, 'theta': theta}
H4 = HillComponent(-1, **parameterDict)  # Test construction by dictionary

# check function calls
print(H1(x0))
print(H2(x0, [delta, n]))
print(H3(x0, [ell, delta, theta, n]))

# check derivative function calls
print(H1.dx(x0))
print(H2.dx(x0, [delta, n]))
print(H3.dx(x0, [ell, delta, theta, n]))

print(H1.dx2(x0, []))
print(H2.dx2(x0, [delta, n]))
print(H3.dx2(x0, [ell, delta, theta, n]))

print(H1.dn(x0))
print(H2.dn(x0, [delta, n]))
print(H2.diff(1, x0, [delta, n]))
print(H3.dn(x0, [ell, delta, theta, n]))
print(H3.diff(1, x0, [ell, delta, theta, n]))

print(H1.dndx(x0))
print(H2.dndx(x0, [delta, n]))
print(H3.dndx(x0, [ell, delta, theta, n]))
