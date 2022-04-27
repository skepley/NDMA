"""
Search for saddle-node bifurcations in the EMT model

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 11/17/2021
"""
from models.EMT_model import *
from saddle_finding_functionalities import *
from create_dataset import *

gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# load the dataset of candidates produced by DSGRN
dataFile = 'dataset_EMT.npz'
n_sample = 2
emtData = np.load(dataFile)
emtRegions = emtData['parameter_region']
monostableIdx = [idx for idx in range(len(emtRegions)) if emtRegions[idx] == 0]
bistableIdx = [idx for idx in range(len(emtRegions)) if emtRegions[idx] == 1]
emtParameters = emtData['data'].transpose()  # transpose to make into an arrow of row vectors.
monostableParameters = emtParameters[monostableIdx]
bistableParameters = emtParameters[bistableIdx]

# Saddle node bifurcation search
SNB = SaddleNode(f)
p = bistableParameters[7]
snb = []
ds = []
dsMinimum = []

data_subsample, region_subsample, coefs = subsample(dataFile, n_sample)
a = data_subsample

n_monostable_region = 0
n_monostable_with_saddle = 0
wrong_parity_monostable = 0
n_monostable_without_saddle = 0
n_monostable_bad_candidate = 0
n_bistable = 0
n_bistable_with_saddle = 0
wrong_parity_bistable = 0
n_bistable_without_saddle = 0
n_bistable_bad_candidate = 0

for d in range(0, n_sample):
    p = data_subsample[:,d]
    region_j = region_subsample[d]
    SNParameters, badCandidates = saddle_node_search(f, [1, 10, 20, 30, 40, 50, 75, 100], p, ds, dsMinimum, maxIteration=100, gridDensity=5, bisectionBool=True)
    if SNParameters and SNParameters != 0:
        if region_j == 1:
            if len(SNParameters) == 1 or len(SNParameters) == 3:
                n_monostable_with_saddle = n_monostable_with_saddle + 1
            else:
                wrong_parity_monostable = wrong_parity_monostable + 1
        else:
            if len(SNParameters) == 2:
                n_bistable_with_saddle = n_bistable_with_saddle + 1
            else:
                wrong_parity_bistable = wrong_parity_bistable + 1
    elif badCandidates and badCandidates != 0:
        if region_j == 1:
            n_monostable_bad_candidate = n_monostable_bad_candidate + 1
        else:
            n_bistable_bad_candidate = n_bistable_bad_candidate + 1
    else:
        if region_j == 4:
            n_monostable_without_saddle = n_monostable_without_saddle + 1
        else:
            n_bistable_without_saddle = n_bistable_without_saddle + 1
# SNB found for hill coeffficients ~ 7.27998 and 25.01123