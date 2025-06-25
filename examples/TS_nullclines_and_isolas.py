import numpy as np
import matplotlib.pyplot as plt

from ndma.hill_model import is_vector
from ndma.basic_models.TS_model import ToggleSwitch

"""
We plot the nullclines of the Toggle Switch
Then, we computer all the equilibria of the Toggle Switch for all Hill coefficients in an interval and plot the result.
For the appropriate parameters, we can detect both hysterisis w.r.t. the Hill coefficient and isolas.
"""

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])

parameter = np.array([0.25, 1, 1, 0.25, 1])
hill = [2.5, 3.8, 7]
for h in hill:
    x1, y1, x2, y2 = f.plot_nullcline(h, parameter, domainBounds=((0, 2), (0, 2)))
    plt.plot(x1, y1, 'g-')
    plt.plot(y2, x2, 'r-')
    title = 'nullclines_'+str(int(h))
    plt.savefig(title)
    plt.show()

parameter = [0.9243, 0.0506, 0.8125, 0.0779, 0.8161]
hill = np.linspace(10, 80, 800)
fig, ax = plt.subplots()
for h in hill:
    eqs = f.global_equilibrium_search(10, h, parameter)
    if is_vector(eqs):
        ax.plot(eqs[0], h, 'b.')
    else:
        for eq in eqs:
            ax.plot(eq[0], h, 'b.')
plt.savefig('hysteresis')
plt.show()


parameter = [0.6470, 0.3279, 0.9445, 0.5301, 0.3908]
hill = np.linspace(10, 80, 800)
fig, ax = plt.subplots()
for h in hill:
    eqs = f.global_equilibrium_search(10, h, parameter)
    if is_vector(eqs):
        ax.plot(eqs[0], h, 'b.')
    else:
        for eq in eqs:
            ax.plot(eq[0], h, 'b.')
plt.savefig('isolas')
plt.show()
