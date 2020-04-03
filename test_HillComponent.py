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
x0 = np.array([4, 3])

# test Hill component code
parm = np.array([ell, theta, delta, n], dtype=float)
H = HillComponent(-1, ell=ell, theta=theta, delta=delta, hillCoefficient=n)
