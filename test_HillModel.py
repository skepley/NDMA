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
    Date: 4/6/20; Last revision: 4/6/20
"""

import numpy as np
import matplotlib.pyplot as plt
from hill_model import HillComponent, HillCoordinate, HillModel


def toggle_switch(gamma, parameter):
    """Defines the vector field for the toggle switch example"""

    # define Hill system for toggle switch
    return HillModel(gamma, parameter, [[-1], [-1]], [[1], [1]], [[1], [0]])


# set some parameters to test using MATLAB toggle switch for ground truth
gamma = np.array([1, 1], dtype=float)
# p1 = np.array([1, 5, 3, 4.1], dtype=float)
# p2 = np.array([1, 6, 3, 4.1], dtype=float)
p1 = np.array([1, 5, 3], dtype=float)
p2 = np.array([1, 6, 3], dtype=float)
parm = np.array([4.1,4.1])
x0 = np.array([4, 3])



# test Hill model code
ts = toggle_switch(gamma, [p1, p2])
print(ts(x0))
# verify that ts2.dx(x0) matches MATLAB - DONE

# test Hill model equilibrium finding
eq = ts.find_equilibria(parm, 10)
print(eq)
# added vectorized evaluation of Hill Models - DONE


# plot nullclines and equilibria
plt.close('all')
Xp = np.linspace(0, 10, 100)
Yp = np.linspace(0, 10, 100)
Z = np.zeros_like(Xp)

N1 = ts.coordinates[0](np.row_stack([Z, Yp])) / gamma[0]  # f1 = 0 nullcline
N2 = ts.coordinates[1](np.row_stack([Xp, Z])) / gamma[1]  # f2 = 0 nullcline

plt.figure()
plt.scatter(eq[0, :], eq[1, :])
plt.plot(Xp, N2)
plt.plot(N1, Yp)
