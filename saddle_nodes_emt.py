"""
Search for saddle-node bifurcations in the EMT model
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 11/17/2021
"""
from models.EMT_model import *
gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# load the dataset of candidates produced by DSGRN
dataFile = 'dataset_EMT.npz'
emtData = np.load(dataFile)
emtRegions = emtData['parameter_region']
monostableIdx = [idx for idx in range(len(emtRegions)) if emtRegions[idx] == 0]
bistableIdx = [idx for idx in range(len(emtRegions)) if emtRegions[idx] == 1]
emtParameters = emtData['data'].transpose()  # transpose to make into an arrow of row vectors.
monostableParameters = emtParameters[monostableIdx]
bistableParameters = emtParameters[bistableIdx]


# # Equilibria search in monostable region
# hill = 5  # an arbitrary Hill coefficient for testing
# badCandidates = []
# for (idx, p) in enumerate(monostableParameters):
#     eq = f.find_equilibria(3, hill, p)
#     nEq = np.shape(eq)[0]
#     print('Parameter: {0}, Equilibria found: {1}'.format(idx, nEq))
#     if nEq != 1:
#         badCandidates += (idx, p)

# # Equilibria search in bistable region
# hill = [2, 5, 10, 20, 30]  # some arbitrary Hill coefficients for a line search
# badCandidates = []
# for (idx, p) in enumerate(bistableParameters[:10]):
#     nEq = []
#     for d in hill:
#         eq = f.find_equilibria(3, d, p)
#         nEq.append(np.shape(eq)[0])
#     print('Parameter: {0}, Equilibria found: {1}'.format(idx, nEq))
# # Parameter 7 returns [1, 1, 3, 3, 1]

# Saddle node bifurcation search
SNB = SaddleNode(f)
p = bistableParameters[7]
snb = []
for d in range(5, 11):
    snb_d = SNB.find_saddle_node(0, d, p)
    if len(snb_d) > 0:
        snb.append(snb_d[0])
        print('SNB found: {0}'.format(snb_d))
# SNB found for hill coeffficients ~ 7.27998 and 25.01123