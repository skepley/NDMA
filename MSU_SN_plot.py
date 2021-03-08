"""
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 7/6/20; Last revision: 7/6/20
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from hill_model import *
from models import ToggleSwitch
plt.close('all')
# decay = np.array([np.nan, np.nan], dtype=float)  # gamma
decay = np.array([np.nan, np.nan])
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (e# ll_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f0 = f.coordinates[0]
f1 = f.coordinates[1]

hill = 5
p = np.array([1, 1, 5, 3, 1, 1, 6, 3])

fig = plt.figure(tight_layout=True, figsize=(9., 3.))
fig.add_subplot(1, 3, 1)
f.plot_nullcline(hill, p)
plt.title('Bistability')

fig.add_subplot(1, 3, 2)
f.plot_nullcline(3.17, p)
plt.title('Saddle Node Bifurcation')


fig.add_subplot(1, 3, 3)
f.plot_nullcline(2.5, p)
plt.title('Monostability')
stopHere

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
