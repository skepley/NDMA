"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/3/20; Last revision: 4/3/20
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from math import log
from hill_model import HillComponent

# set some parameters to test using MATLAB toggle switch for ground truth
ell = 1
theta = 3
delta = 5
n = 4.1
x0 = np.array([3,4])
# x0 = 3

# test Hill component code
parm = np.array([ell, theta, delta, n], dtype=float)
H1 = HillComponent(-1, ell=ell, theta=theta, delta=delta, hillCoefficient=n)  # A function of x with all parameters fixed
H2 = HillComponent(-1, ell=ell, theta=theta)  # A function of x with callable variable parameters: {delta, n}
H3 = HillComponent(-1)  # A function of x with all parameters callable

# check function calls
print(H1(x0))
print(H2(x0, [delta, n]))
print(H3(x0, [ell, delta, theta, n]))

# check derivative function calls
print(H1.dx(x0))
print(H2.dx(x0, [delta, n]))
print(H3.dx(x0, [ell, delta, theta, n]))

print(H1.dx(x0, [], 2))
print(H2.dx(x0, [delta, n], 2))
print(H3.dx(x0, [ell, delta, theta, n], 2))

print(H1.dn(x0))
print(H2.dn(x0, [delta, n]))
print(H3.dn(x0, [ell, delta, theta, n]))

print(H1.dndx(x0))
print(H2.dndx(x0, [delta, n]))
print(H3.dndx(x0, [ell, delta, theta, n]))