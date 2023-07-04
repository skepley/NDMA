"""
Create the plot for Application #1 in section 6 of the paper

    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 3/23/22; Last revision: 3/23/22
"""
import matplotlib.pyplot as plt
from saddle_finding_functionalities import *

# set up the model
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)

# identify the saddle node parameter values
hillRange = np.arange(2, 5)
ds = 0.1
dsMinimum = 0.001
SNB, BC = saddle_node_search(f, hillRange, p0, ds, dsMinimum, gridDensity=5)

# plot nullclines
fig, (ax2, ax1, ax0) = plt.subplots(nrows=1, ncols=3,
                                    figsize=(12, 4))
# ax.set(xlabel=r'$x_1(d)$', ylabel=r'$d$')

ax2.set_title(r'$d = 2.5$')
f.plot_nullcline(2.5, p0, ax=ax2)

ax1.set_title(r'$d \approx {0}$'.format(np.round(SNB[0][1], 4)))
f.plot_nullcline(SNB[0][1], p0, ax=ax1)

ax0.set_title(r'$d = 4.5$')
f.plot_nullcline(4.5, p0, ax=ax0)

fig.suptitle('Toggle Switch nullclines along a Hill path')
plt.show()
fig.savefig('nullclines.png')

I won 43 tournament-size chess sets (nooards) at auction for $14… what do I do with them ? ?