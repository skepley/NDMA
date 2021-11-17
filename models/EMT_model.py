"""
An implementation of the 6 node EMT network as a Hill model
    Output: output
    Other files required: none
    See also:
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 1/15/2021
"""

from hill_model import *
from saddle_node import *

class EMT(HillModel):
    """Six-dimensional EMT model construction inherited as a HillModel where each node has free Hill coefficients. This
     has a total of 12 edges and 54 parameters. The nodes are ordered as follows:
    0. TGF_beta
    1. miR200
    2. Snail1
    3. Ovol2
    4. Zeb1
    5. miR34a"""

    def __init__(self, gamma, parameter):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^6 of linear decay rates or NaN if decays are variable parameters.
            parameter - A length-6 list of parameter arrays of size K_i-by-4 where K_i is the number of incoming edges to
             node i. Each row of a parameter array has the form (ell, delta, theta, hill)."""

        productionSigns = [[-1, -1], [-1, -1], [1, -1], [-1], [-1, 1, -1],
                           [-1, -1]]  # length 6 list of production signs for each node
        edgeCounts = [len(sign) for sign in productionSigns]
        productionTypes = [nEdge * [1] for nEdge in edgeCounts]  # all productions are products
        productionIndex = [[0, 1, 3], [1, 2, 4], [2, 0, 5], [3, 4], [4, 1, 2, 3], [5, 2, 4]]
        super().__init__(gamma, parameter, productionSigns, productionTypes,
                         productionIndex)  # define HillModel for toggle switch by inheritance


gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(4)] for k in range(nEdge)]).squeeze() for nEdge in edgeCounts]
f = EMT(gammaVar, parameterVar)
SNB = SaddleNode(f)

gammaValues = np.array([j for j in range(1, 7)])
parmValues = [np.random.rand(*np.shape(parameterVar[node])) for node in range(6)]
x = np.random.rand(6)
p = ezcat(*[ezcat(ezcat(tup[0], tup[1].flatten())) for tup in
            zip(gammaValues, parmValues)])  # this only works when all parameters are variable

print(np.shape(f(x, p)))
print(np.shape(f.dx(x, p)))
print(np.shape(f.diff(x, p)))
print(np.shape(f.dx2(x, p)))
print(np.shape(f.dxdiff(x, p)))
print(np.shape(f.diff2(x, p)))





