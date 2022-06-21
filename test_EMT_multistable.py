"""
Search for saddle-node bifurcations in the EMT model

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 11/17/2021
"""
from models.EMT_model import *
from saddle_finding_functionalities import *
from create_dataset import *
import sys
from scipy.stats import chi2_contingency

gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# load the dataset of candidates produced by DSGRN
dataFile = 'dataset_bistable_EMT.npz'
file_storing = 'chi_test_EMT_Bedlewo.npz'
n_sample = 100

emtData = np.load(dataFile)
emtRegions = emtData['parameter_region']
Idx = [idx for idx in range(len(emtRegions)) if emtRegions[idx] == 0]
emtParameters = emtData['data'].transpose()  # transpose to make into an arrow of row vectors.
Parameters = emtParameters[Idx]

random_index_multistable = random.sample(Idx, n_sample)
all_index = set(random_index_multistable)
n_sample = len(all_index)
data_subsample = emtParameters[all_index].transpose()
region_subsample = emtRegions[all_index]
n_sample = len(all_index)

for d in range(0, n_sample):
    p = data_subsample[:, d]
    region_j = region_subsample[d]
    eqs = f.find_equilibria(3, 50, p)
    n_eqs = len(eqs)
    printing_statement = 'Completion: ' + str(d + 1) + ' out of ' + str(n_sample) + '\n n_eqs = ' + n_eqs
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()
