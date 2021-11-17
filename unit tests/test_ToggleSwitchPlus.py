"""
Testing functionality for the ToggleSwitch with self edges allowed.

    Other files required: hill_model
    See also: test_ToggleSwitch
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 7/9/2020 
"""
from hill_model import *
from models.TSPlus_model import ToggleSwitchPlus

# TESTING FOR TOGGLE SWITCH PLUS
# ============= set up the toggle switch example to test on =============
nCoordinate = 2
gamma = np.array([1, 1], dtype=float)
hill = 4.1

# TURN ON THE SECOND SELF ACTIVATION
componentParmValues = [np.array([1, 5, 3], dtype=float),
                       np.array([[1, 6, 3], [1, 6, 3]], dtype=float)]
selfInteractions = [0, -1]
parameter1 = [np.copy(cPValue) for cPValue in componentParmValues]
for parm in parameter1:
    parm[:] = np.nan

# # TURN ON BOTH SELF ACTIVATIONS
# componentParmValues = [np.array([[1, 5, 3], [1, 5, 3]], dtype=float),
#                        np.array([[1, 6, 3], [1, 6, 3]], dtype=float)]
# selfInteractions = [-1, -1]
# parameter1 = [np.copy(cPValue) for cPValue in componentParmValues]
# for parm in parameter1:
#     parm[:] = np.nan


gammaVar = np.array([np.nan, np.nan])  # set both decay rates as variables
f = ToggleSwitchPlus(gammaVar, parameter1, selfInteractions)
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H11 = f1.productionComponents[0]
# H12 = f1.productionComponents[1]
H21 = f2.productionComponents[0]
H22 = f2.productionComponents[1]

p1 = componentParmValues[0].flatten()
p2 = componentParmValues[1].flatten()
componentParmVectors = [p1, p2]
p = ezcat(
    *[ezcat(*tup) for tup in zip(gamma, componentParmVectors)])  # this only works when all parameters are variable
x = np.array([2, 3])
