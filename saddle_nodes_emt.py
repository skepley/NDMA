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

for idx in range(10):
    p = monostableParameters[idx]
    eq = f.find_equilibria(3, p)
    print(eq)

