"""
One line description of what the script performs (H1 line)
  
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 7/20/2020 
"""
import numpy as np

from ndma.hill_model import ezcat
from ndma.examples.Network12_model import Network12

# ============= set up the a Network12 instance =============
nCoordinate = 3
gamma = np.array([1, 1, 1], dtype=float)
hill = 4.1

componentParmValues = [np.array([[1, 2, 3], [3, 2, 1], [2, 3, 1]], dtype=float),
                       np.array([[3, 2, 1], [2, 1, 3]], dtype=float),
                       np.array([3, 1, 2], dtype=float)]
parameter1 = [np.copy(cPValue) for cPValue in componentParmValues]
for parm in parameter1:
    parm[:] = np.nan

gammaVar = np.array([np.nan, np.nan, np.nan])  # set all decay rates as variables
f = Network12(gammaVar, parameter1)
f0 = f.coordinates[0]
f1 = f.coordinates[1]
f2 = f.coordinates[2]

p0 = componentParmValues[0].flatten()
p1 = componentParmValues[1].flatten()
p2 = componentParmValues[2].flatten()

componentParmVectors = [p0, p1, p2]
p = ezcat(
    *[ezcat(*tup) for tup in zip(gamma, componentParmVectors)])  # this only works when all parameters are variable
x = np.array([1., 2, 3])
