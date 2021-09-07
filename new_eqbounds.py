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

from hill_model import *
from models.TS_model import ToggleSwitch

# set some parameters to test using MATLAB toggle switch for ground truth
# decay = np.array([np.nan, np.nan], dtype=float)  # gamma
decay = np.array([1, 1])
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (e# ll_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
H1 = f.coordinates[0].components[0]
H2 = f.coordinates[1].components[0]
# SN = SaddleNode(f)

# p = np.random.random_sample([1, 8])
p = np.arange(1, 9)
p1 = p[0:4]
p2 = p[4:]
x1Bounds = H1.image(p1)
x2Bounds = H2.image(p2)
x1New = np.array([H1(x1, p1) for x1 in x1Bounds])
x2New = np.array([H2(x2, p2) for x2 in x2Bounds])


def update_bounds(p, x1Bounds, x2Bounds):
    """Set new bounds for toggle switch equilibirum box """
    p1 = p[0:4]
    p2 = p[4:]
    x1New = np.array([H1(x1, p1) for x1 in x1Bounds])
    x2New = np.array([H2(x2, p2) for x2 in x2Bounds])
    return np.sort(x1New), np.sort(x2New)

plt.figure()
f.plot_nullcline(p, domainBounds=(tuple(x1Bounds), tuple(x2Bounds)))

# for iter in range(10):
#     print(x1Bounds, x2Bounds)
#     x1Bounds, x2Bounds = update_bounds(p, x1Bounds, x2Bounds)
#
# print(x1Bounds, x2Bounds)
#
