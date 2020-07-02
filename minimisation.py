"""
Tests to learn about pythin and minimization

    Author: Elena Queirolo
    email: elena.queirolo@rutgers.edu
"""
import numpy as np
import random
from scipy import optimize
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


def rosen(x):
    """The Rosenbrock function """
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def silly_function(x):
    return math.sin(sum((x - 56) ** 2))


x0 = np.array([1.3, 7.9, 0.8, 1.9, 1.2])
res = minimize(silly_function, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)
