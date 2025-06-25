"""
PLot the hysteresis figure from the Toggle Switch section of the paper.

    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 3/22/22; Last revision: 3/22/22
"""
from ndma.saddle_finding_functionalities import *

# set up the model
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
p_hyst = np.array([1, 0.92436706, 0.05063294, 1, 0.81250005, 0.07798304, 0.81613, 1])  # hysteresis parameter from the paper


# identify the saddle node parameter values
hillRange = np.arange(1, 70, 5)
ds = 0.1
dsMinimum = 0.001
SNB, BC = saddle_node_search(f, hillRange, p_hyst, ds, dsMinimum, gridDensity=5)


# plot 1
fig, ax = plt.subplots()
ax.set(xlabel=r'$x_1(d)$', ylabel=r'$d$')
ds = 0.1  # arc length parameter
h0 = 80
h0Target = SNB[0][1]
eq1 = f.find_equilibria(10, h0, p_hyst)
eq1_Branch, hill1_Branch = continue_equilibrium(f, eq1, h0, h0Target, p_hyst, ds, maxIteration=2000)
xHat = np.squeeze(eq1_Branch[:, 0])
ax.plot(xHat, hill1_Branch, color='black')

# plot 2
ds = 0.1  # arc length parameter
h0 = 50
h0Target = SNB[0][1]
eq2 = f.find_equilibria(10, h0, p_hyst)[1]
eq2_Branch, hill2_Branch = continue_equilibrium(f, eq2, h0, h0Target, p_hyst, ds, maxIteration=2000)
xHat = np.squeeze(eq2_Branch[:, 0])
ax.plot(xHat, hill2_Branch, color='black')

# plot 3
ds = 0.1  # arc length parameter
h0 = 50
h0Target = SNB[1][1]
eq3 = f.find_equilibria(10, h0, p_hyst)[1]
eq3_Branch, hill3_Branch = continue_equilibrium(f, eq3, h0, h0Target, p_hyst, ds, maxIteration=2000)
xHat = np.squeeze(eq3_Branch[:, 0])
ax.plot(xHat, hill3_Branch, color='black')

# plot 4
ds = 0.1  # arc length parameter
h0 = 10
h0Target = SNB[1][1]
eq4 = f.find_equilibria(10, h0, p_hyst)
eq4_Branch, hill4_Branch = continue_equilibrium(f, eq4, h0, h0Target, p_hyst, ds, maxIteration=2000)
xHat = np.squeeze(eq4_Branch[:, 0])
ax.plot(xHat, hill4_Branch, color='black')
plt.show()

fig.savefig('figure_hysteresis.png')
