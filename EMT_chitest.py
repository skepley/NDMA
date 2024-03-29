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
from time import time

t = time()

gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# load the dataset of candidates produced by DSGRN
dataFile = 'dataset_multistable_EMT.npz'
file_storing = 'chi_test_EMT_15July22.npz'
n_sample = 1000
data_subsample, region_subsample, coefs = subsample(dataFile, n_sample, [0, 1])

#emtData = np.load(dataFile)
#data_subsample = emtData['data']
#emtRegions = emtData['parameter_region']
#region_subsample = emtRegions
#n_sample = len(emtRegions)

#n_sample = 1600
#data_subsample, region_subsample, coefs = subsample(dataFile, n_sample)
#a = data_subsample

# Saddle node bifurcation search
# SNB = SaddleNode(f)
# p = bistableParameters[7]
# snb = []
ds = []
dsMinimum = []

# data_subsample, region_subsample, coefs = subsample(dataFile, n_sample)
# a = data_subsample

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
    p = data_subsample[:, d]
    region_j = region_subsample[d]
    SNParameters, badCandidates = saddle_node_search(f, [1, 10, 20, 30, 40, 50, 75, 100], p, ds, dsMinimum, maxIteration=100, gridDensity=3, bisectionBool=True)
    if SNParameters and SNParameters != 0:
        if region_j == 0:
            n_monostable_with_saddle = n_monostable_with_saddle + 1
        else:
            n_bistable_with_saddle = n_bistable_with_saddle + 1

    elif badCandidates and badCandidates != 0:
        if region_j == 0:
            n_monostable_bad_candidate = n_monostable_bad_candidate + 1
        else:
            n_bistable_bad_candidate = n_bistable_bad_candidate + 1
    else:
        if region_j == 0:
            n_monostable_without_saddle = n_monostable_without_saddle + 1
        else:
            n_bistable_without_saddle = n_bistable_without_saddle + 1
    
    printing_statement = 'Completion: ' + str(d+1) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()


try:
    data = np.load(file_storing, allow_pickle=True)

    v_monostable_region = data.f.v_monostable_region
    v_monostable_with_saddle = data.f.v_monostable_with_saddle
    v_monostable_without_saddle = data.f.v_monostable_without_saddle
    v_monostable_bad_candidate = data.f.v_monostable_bad_candidate
    v_wrong_parity_monostable = data.f.v_wrong_parity_monostable
    v_bistable = data.f.v_bistable
    v_bistable_with_saddle = data.f.v_bistable_with_saddle
    v_bistable_without_saddle = data.f.v_bistable_without_saddle
    v_bistable_bad_candidate = data.f.v_bistable_bad_candidate
    v_sample = data.f.v_sample
    v_wrong_parity_bistable = data.f.v_wrong_parity_bistable

    v_monostable_region = np.append(v_monostable_region, np.array([n_monostable_region]))
    v_monostable_with_saddle = np.append(v_monostable_with_saddle, np.array([n_monostable_with_saddle]))
    v_wrong_parity_monostable = np.append(v_wrong_parity_monostable, np.array([wrong_parity_monostable]))
    v_monostable_without_saddle = np.append(v_monostable_without_saddle, np.array([n_monostable_without_saddle]))
    v_monostable_bad_candidate = np.append(v_monostable_bad_candidate, np.array([n_monostable_bad_candidate]))
    v_bistable = np.append(v_bistable, np.array([n_bistable]))
    v_bistable_with_saddle = np.append(v_bistable_with_saddle, np.array([n_bistable_with_saddle]))
    v_bistable_without_saddle = np.append(v_bistable_without_saddle, np.array([n_bistable_without_saddle]))
    v_bistable_bad_candidate = np.append(v_bistable_bad_candidate, np.array([n_bistable_bad_candidate]))
    v_wrong_parity_bistable = np.append(v_wrong_parity_bistable, np.array([wrong_parity_bistable]))
    v_sample = np.append(v_sample, np.array([n_sample]))

except:
    v_monostable_region = np.array([n_monostable_region])
    v_monostable_with_saddle = np.array([n_monostable_with_saddle])
    v_wrong_parity_monostable = np.array([wrong_parity_monostable])
    v_monostable_without_saddle = np.array([n_monostable_without_saddle])
    v_monostable_bad_candidate = np.array([n_monostable_bad_candidate])
    v_bistable = np.array([n_bistable])
    v_bistable_with_saddle = np.array([n_bistable_with_saddle])
    v_wrong_parity_bistable = np.array([wrong_parity_bistable])
    v_bistable_without_saddle = np.array([n_bistable_without_saddle])
    v_bistable_bad_candidate = np.array([n_bistable_bad_candidate])
    v_sample = np.array([n_sample])

np.savez(file_storing,
         v_monostable_region=v_monostable_region, v_monostable_with_saddle=v_monostable_with_saddle,
         v_monostable_without_saddle=v_monostable_without_saddle,
         v_monostable_bad_candidate=v_monostable_bad_candidate, v_bistable=v_bistable, v_bistable_with_saddle=v_bistable_with_saddle,
         v_bistable_without_saddle=v_bistable_without_saddle, v_bistable_bad_candidate=v_bistable_bad_candidate, v_sample=v_sample)

data = np.load(file_storing)

mat_for_chi_test = np.array([[np.sum(v_bistable_with_saddle), np.sum(v_bistable_without_saddle)], [np.sum(v_monostable_with_saddle), np.sum(v_monostable_without_saddle)]])

print('\nCorrelation matrix\n')
print(mat_for_chi_test)

unused, p, a, b = chi2_contingency(mat_for_chi_test)

if p <= 0.05:
    print('We reject the null hypothesis: there is correlation between saddles and center region\n')
else:
    print('We cannot reject the null hypothesis: there is NO proven correlation between saddles and center region\n')

print('p-value = ', p)
toc = time()
print(toc-t + ' sec Elapsed')

print('It is the end!')