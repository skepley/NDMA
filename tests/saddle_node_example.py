"""
Testing and analysis for SaddleNode and ToggleSwitch classes

    Other files required: hill_model, saddle_node, models

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 4/24/2020 
"""

from ndma.hill_model import *
from ndma.model import Model
from ndma.examples.TS_model import ToggleSwitch
from ndma.bifurcation.saddlenode import SaddleNode
import matplotlib.pyplot as plt


np.set_printoptions(precision=2, floatmode='maxprec_equal')

# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.productionComponents[0]
H2 = f2.productionComponents[0]

p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
SN = SaddleNode(f)

# ==== find saddle node for a parameter choice
rho = 4.1
p = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)


# x0Sol, v0Sol, rhoSol = [u0Sol.x[idx] for idx in [[0, 1], [2, 3], [4]]]
# # compare to rhoSol = [ 4.55637172,  2.25827744,  0.82199933, -0.56948846,  3.17447061]

# plot nullclines and equilibria
plt.close('all')
plt.figure()
f.plot_nullcline(rho, p)
plt.title('Initial parameters: \n' + np.array2string(ezcat(rho, p)))

fig = plt.figure(tight_layout=True, figsize=(15., 9.))
allSol = []
fullParameter = ezcat(rho, p)

for j in range(9): # put back range 9 for test
    fig.add_subplot(3, 3, j + 1)
    jSearchNodes = np.linspace(fullParameter[j] / 10, 10 * fullParameter[j], 25)
    print(jSearchNodes)
    jSols = SN.find_saddle_node(j, rho, p, freeParameterValues=jSearchNodes)
    print(j, jSols)
    allSol.append(jSols)
    for sol in jSols:
        pSol = fullParameter.copy()
        pSol[j] = sol
        f.plot_nullcline(pSol)
    plt.title('parameter: {0}'.format(j))

plt.close('all')
plt.figure()
for hillC in np.linspace(1, 3, 5):
    plt.figure()
    f.plot_nullcline(hillC, p)

# ==== This one finds a pitchfork bifurcation instead
p1 = np.array([1, 1, 5, 3, 1, 1, 5, 3], dtype=float)

n0 = 3.179
eq1 = f.find_equilibria(4, n0, p1)
n1Sol = SN.find_saddle_node(0, n0, p1)

# plot nullclines
plt.figure()
f.plot_nullcline(n0, p1)
plt.title('p = {0}; n = {1}'.format(p1, n0))
plt.show()
plt.figure()
f.plot_nullcline(n1Sol, p1)
plt.title('p = {0}; n = {1}'.format(p1, n1Sol))
plt.figure()
epsilon = 0.01
f.plot_nullcline(n1Sol+epsilon, p1)
plt.title('p = {0}; n = {1}'.format(p1, n1Sol+epsilon))

plt.show()
