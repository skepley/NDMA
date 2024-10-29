"""
This code creates the heat map, as presented in the article.
The heat map indicates the value of the Hill coefficient in which a saddle node is taking place depending on the parameter.
It also consider the parameter projection into [0,3]x[0,3] thanks to the DSGRN region definition
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from saddle_finding_functionalities import saddle_node_search
from saddle_node import SaddleNode
from toggle_switch_heat_functionalities import parameter_to_alpha_beta, parameter_to_DSGRN_coord, dsgrn_heat_plot, \
    dsgrn_plot
from models.TS_model import ToggleSwitch
from create_dataset import create_dataset_ToggleSwitch, subsample

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# use dataset creation
# size of the sample
n_sample = 4 * 10 ** 4  # testing on 3, final run on 4
file_name = 'TS_data_100000.npz'
try:
    np.load(file_name)
except FileNotFoundError:
    n = 100000
    create_dataset_ToggleSwitch(100000, file_name)

file_storing = 'heat_map_2024.npz'

data_subsample, region_subsample = subsample(file_name, n_sample)
# a = np.transpose(data_subsample)
a = data_subsample
u, v = parameter_to_DSGRN_coord(a)


# test for nice heat map pictures
alpha1, beta1, alpha2, beta2 = parameter_to_alpha_beta(a)
alpha1 = np.sort(alpha1)
alpha2 = np.sort(alpha2)
ninety_percentile = int(np.ceil(len(alpha1)*0.9))
alphaMax = np.array([alpha1[ninety_percentile], alpha2[ninety_percentile]])

parameter_full = np.empty(shape=[0, 5])
lowest_hill = np.empty(0)
bad_parameters = np.empty(shape=[0, 5])
bad_candidates = []
boring_parameters = np.empty(shape=[0, 5])
multiple_saddles = np.empty(shape=[0, 5])
for j in range(n_sample):  # range(n_sample):
    a_j = a[j, :]
    ds = 0.01
    dsMinimum = 0.005
    SNParameters, badCandidates = saddle_node_search(f, [1, 5, 10, 40, 80], a_j, ds, dsMinimum, maxIteration=100,
                                                     gridDensity=5, bisectionBool=True)
    if SNParameters and SNParameters != 0:
        for k in range(len(SNParameters)):
            # print('Saddle detected')
            if k == 0:
                parameter_full = np.append(parameter_full, [a_j], axis=0)
                lowest_hill = np.append(lowest_hill, SNParameters[k][1])
            if k > 0:
                print('\nMore than one saddle detected!')
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates and badCandidates != 0:
        print('\nA bad parameter')
        bad_parameters = np.append(bad_parameters, [a_j], axis=0)
        bad_candidates.append(badCandidates)
    printing_statement = 'Completion: ' + str(j) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

    if SNParameters == 0 and badCandidates == 0:
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)

'''
# uncomment to save the results
np.savez('heat_map_data',
         u=u, v=v, a=a, parameter_full=parameter_full, lowest_hill=lowest_hill, bad_parameters=bad_parameters,
         bad_candidates=bad_candidates, boring_parameters=boring_parameters, n_sample=n_sample,
         multiple_saddles=multiple_saddles)

data = np.load('heat_map_data.npz', allow_pickle=True)

bad_parameters = data.f.bad_parameters
boring_parameters = data.f.boring_parameters
n_sample = data.f.n_sample
parameter_full = data.f.parameter_full
lowest_hill = data.f.lowest_hill
multiple_saddles = data.f.multiple_saddles
'''

print('\nNumber of bad candidates', len(bad_parameters), 'out of ', n_sample)
print('Number of boring candidates', len(boring_parameters), 'out of ', n_sample)
print('Number of saddles', len(parameter_full), 'out of ', n_sample)
print('Number of parameters with multiple saddles', len(multiple_saddles), 'out of ', n_sample)


plt.figure()
dsgrn_heat_plot(parameter_full, np.minimum(lowest_hill, 100), alphaMax=alphaMax)
plt.title('dsgrn_heat_plot')
plt.savefig('dsgrn_heat_plot.pdf')

plt.figure()
dsgrn_plot(a, color='tab:blue', alphaMax=alphaMax, alpha=1)
dsgrn_plot(parameter_full, color='tab:green', alphaMax=alphaMax)
if len(multiple_saddles) > 0:
    dsgrn_plot(multiple_saddles, color='tab:orange', alphaMax=alphaMax)
if len(bad_parameters) > 0:
    dsgrn_plot(bad_parameters, color='tab:red', alphaMax=alphaMax)
#plt.show()
plt.savefig('all_results.pdf')

print('It is the end!')
