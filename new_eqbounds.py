"""
One line description of what the script performs (H1 line)
  
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 7/6/20; Last revision: 7/6/20
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from hill_model import *
from models import ToggleSwitch

# set some parameters to test using MATLAB toggle switch for ground truth
# decay = np.array([np.nan, np.nan], dtype=float)  # gamma
decay = np.array([np.nan, np.nan])
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (e# ll_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f0 = f.coordinates[0]
f1 = f.coordinates[1]

hill = 3.25
# p = np.random.randint(3, 10, 8)
# p = np.arange(2, 10)
p = np.array([1, 1, 5, 3, 1, 1, 6, 3])
# p = np.array([3, 1, 4, 3, 4, 1, 5, 3])


# fullParm = f.parse_parameter(hill, p)
# P0, P1 = f.unpack_variable_parameters(fullParm)  # unpack variable parameters by component
# g0, p0 = f0.parse_parameters(P0)
# g1, p1 = f1.parse_parameters(P1)
# x1Bounds = H1.image(p0[0])
# x2Bounds = H2.image(p1[0])
# x1New = np.array([H1(x1, p1) for x1 in x1Bounds])
# x2New = np.array([H2(x2, p2) for x2 in x2Bounds])

# def bootstrap_map(*parameter):
#     """Return the bootstrap map for the toggle switch, Phi: R^4 ---> R^4 which iterated to bound equilibrium enclosures"""
#
#     fullParm = f.parse_parameter(
#         *parameter)  # concatenate all parameters into a vector with hill coefficients sliced in
#     P0, P1 = parameterByCoordinate = f.unpack_variable_parameters(fullParm)  # unpack variable parameters by component
#     g0, p0 = f.coordinates[0].parse_parameters(P0)
#     g1, p1 = f.coordinates[1].parse_parameters(P1)
#
#     def H0(x):
#         """Evaluate the function from R^2 to R defined by the first and 3rd components of Phi"""
#         return (1 / g0) * f.coordinates[0].components[0](x[1], p0[0])
#
#     def H1(x):
#         """Evaluate the function from R^2 to R defined by the second and fourth components of Phi"""
#         return (1 / g1) * f.coordinates[1].components[0](x[0], p1[0])
#
#     def bootstrap(u):
#         alpha, beta = np.split(u, 2)
#         alphaNew = np.array([H0(beta), H1(beta)])
#         betaNew = np.array([H0(alpha), H1(alpha)])
#         return ezcat(alphaNew, betaNew)
#
#     return bootstrap

parameterByCoordinate = f.unpack_variable_parameters(
    f.parse_parameter(hill, p))  # unpack variable parameters by component
eqBound = np.array(list(map(lambda f_i, parm: f_i.eq_interval(parm), f.coordinates, parameterByCoordinate)))
u0, uB = f.bootstrap_enclosure(hill, p)
eq = f.find_equilibria(10, hill, p, bootstrap=False)
# print(eq)
eq2 = f.find_equilibria(10, hill, p)
# plot bootstrap boxes
# choose initial condition for Phi
# u0 = np.array([3, 3, 5, 5])

# iterate the bootstrap map to obtain an enclosure
Phi = f.bootstrap_map(hill, p)


def plot_box(u, ax):
    x = np.array([u[0], u[2], u[2], u[0], u[0]])
    y = np.array([u[1], u[1], u[3], u[3], u[1]])
    ax.plot(x, y)


fig = plt.figure()
ax = plt.gca()
f.plot_nullcline(hill, p)
u = u0
plot_box(u0, ax)
for j in range(68):
    u = Phi(u)
    plot_box(u, ax)

ax.set_xlim(u0[::2])
ax.set_ylim(u0[1::2])
# f.plot_nullcline(hill, p, domainBounds=(tuple(x1Bounds), tuple(x2Bounds)))
# u = bootstrap_enclosure(f, hill, p)
#
# print(u)
#
