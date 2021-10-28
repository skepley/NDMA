"""
Function and design testing for the SaddleNode class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/13/20; Last revision: 4/13/20
"""
from hill_model import SaddleNode
from test_HillModel import ToggleSwitch


# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([1, 1], dtype=float)
p1 = np.array([1, 5, 3], dtype=float)
p2 = np.array([1, 6, 3], dtype=float)
x = np.array([4, 3])

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.productionComponents[0]
H2 = f2.productionComponents[0]
n0 = 4.1

SN = SaddleNode(f, unit_phase_condition, diff_unit_phase_condition)
eq = f.find_equilibria(n0, 10)
# v0 = np.array([1, -.7])
v0 = np.array([1, 1])
x0 = eq[:, 1]
u0 = np.concatenate((x0, v0, np.array(n0)), axis=None)

uSol = find_root(SN.zero_map, SN.diff_zero_map, u0, diagnose=True)
print(uSol)
xSol, vSol, nSol = [uSol.x[idx] for idx in [[0, 1], [2, 3], [4]]]

# plot nullclines
plt.close('all')
plt.figure()
f.plot_nullcline(4.1)
plt.figure()
f.plot_nullcline(nSol)