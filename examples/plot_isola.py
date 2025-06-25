"""
PLot the isola figure from the Toggle Switch section of the paper.

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 3/22/22; Last revision: 3/22/22
"""
import matplotlib.pyplot as plt
from ndma.saddle_finding_functionalities import *

# set up the model
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
p_isola = np.array([1, 0.64709401, 0.32790599, 1, 0.94458637, 0.53012047, 0.39085124, 1])  # isola example from the paper


# identify the saddle node parameter values
hillRange = np.arange(1, 70, 5)
ds = 0.1
dsMinimum = 0.001
SNB, BC = saddle_node_search(f, hillRange, p_isola, ds, dsMinimum, gridDensity=5)
fig, ax = plt.subplots()  # set up plot
ax.set(xlabel=r'$x_1(d)$', ylabel=r'$d$')


# plot 1
ds = 0.1  # arc length parameter
h0 = 80
h0Target = 10
eq1 = f.find_equilibria(10, h0, p_isola)
eq1_Branch, hill1_Branch = continue_equilibrium(f, eq1, h0, h0Target, p_isola, ds, maxIteration=2000)
xHat = np.squeeze(eq1_Branch[:, 0])
ax.plot(xHat, hill1_Branch, color='black')

# plot 2
ds = 0.1  # arc length parameter
h0 = 50
h0Target = SNB[0][1]
eq2 = f.find_equilibria(10, h0, p_isola)
for eq in eq2[:2]:
    eq2_Branch, hill2_Branch = continue_equilibrium(f, eq, h0, h0Target, p_isola, ds, maxIteration=2000)
    xHat = np.squeeze(eq2_Branch[:, 0])
    ax.plot(xHat, hill2_Branch, color='black')



# plot 3
ds = 0.1  # arc length parameter
h0 = 50
h0Target = SNB[1][1]
eq3 = f.find_equilibria(10, h0, p_isola)
for eq in eq3[:2]:
    eq3_Branch, hill3_Branch = continue_equilibrium(f, eq, h0, h0Target, p_isola, ds, maxIteration=2000)
    xHat = np.squeeze(eq3_Branch[:, 0])
    ax.plot(xHat, hill3_Branch, color='black')


plt.show()
fig.savefig('figure_isolas.png')
