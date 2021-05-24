from hill_model import *
from saddle_finding import *
from toggle_switch_heat_functionalities import *
import matplotlib.pyplot as plt
import sys
from create_dataset import *

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
n_sample = 10**3
file_name = 'TS_data_100000.npz'
file_storing = 'chi_test_data.npz'

data_subsample, region_subsample, coefs = subsample(file_name, n_sample)
a = data_subsample

n_center_region = 0
n_center_with_saddle = 0
n_center_without_saddle = 0
n_center_bad_candidate = 0
n_donut = 0
n_donut_with_saddle = 0
n_donut_without_saddle = 0
n_donut_bad_candidate = 0

for j in range(n_sample):#range(n_sample):
    a_j = a[:, j]
    region_j = region_subsample[j]
    if region_j is 5:
        n_center_region = n_center_region + 1
    else:
        n_donut = n_donut + 1
    SNParameters, badCandidates = find_saddle_coef(f, [1, 50], a_j)
    if SNParameters and SNParameters is not 0:
        if region_j is 5:
            n_center_with_saddle = n_center_with_saddle + 1
        else:
            n_donut_with_saddle = n_donut_with_saddle + 1
    elif badCandidates and badCandidates is not 0:
        if region_j is 5:
            n_center_bad_candidate = n_center_bad_candidate + 1
        else:
            n_donut_bad_candidate = n_donut_bad_candidate + 1
    else:
        if region_j is 5:
            n_center_without_saddle = n_center_without_saddle + 1
        else:
            n_donut_without_saddle = n_donut_without_saddle + 1
    printing_statement = 'Completion: ' + str(j) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

# print('\nNumber of bad candidates', len(bad_candidates), 'out of ', n_sample)
# print('Number of boring candidates', len(boring_parameters), 'out of ', n_sample)
# print('Number of saddles', len(parameter_full), 'out of ', n_sample)

try:
    data = np.load(file_storing, allow_pickle=True)

    v_center_region = data.f.v_center_region
    v_center_with_saddle = data.f.v_center_with_saddle
    v_center_without_saddle = data.f.v_center_without_saddle
    v_center_bad_candidate = data.f.v_center_bad_candidate
    v_donut = data.f.v_donut
    v_donut_with_saddle = data.f.v_donut_with_saddle
    v_donut_without_saddle = data.f.v_donut_without_saddle
    v_donut_bad_candidate = data.f.v_donut_bad_candidate
    v_sample = data.f.v_sample

    v_center_region = np.append(v_center_region, np.array([n_center_region]))
    v_center_with_saddle = np.append(v_center_with_saddle, np.array([n_center_with_saddle]))
    v_center_without_saddle = np.append(v_center_without_saddle, np.array([n_center_without_saddle]))
    v_center_bad_candidate = np.append(v_center_bad_candidate, np.array([n_center_bad_candidate]))
    v_donut = np.append(v_donut, np.array([n_donut]))
    v_donut_with_saddle = np.append(v_donut_with_saddle, np.array([n_donut_with_saddle]))
    v_donut_without_saddle = np.append(v_donut_without_saddle, np.array([n_donut_without_saddle]))
    v_donut_bad_candidate = np.append(v_donut_bad_candidate, np.array([n_donut_bad_candidate]))
    v_sample = np.append(v_sample, np.array([n_sample]))

except:
    v_center_region = np.array([n_center_region])
    v_center_with_saddle =  np.array([n_center_with_saddle])
    v_center_without_saddle = np.array([n_center_without_saddle])
    v_center_bad_candidate = np.array([n_center_bad_candidate])
    v_donut = np.array([n_donut])
    v_donut_with_saddle = np.array([n_donut_with_saddle])
    v_donut_without_saddle = np.array([n_donut_without_saddle])
    v_donut_bad_candidate = np.array([n_donut_bad_candidate])
    v_sample = np.array([n_sample])

np.savez(file_storing,
         v_center_region=v_center_region, v_center_with_saddle=v_center_with_saddle, v_center_without_saddle=v_center_without_saddle,
         v_center_bad_candidate=v_center_bad_candidate, v_donut=v_donut, v_donut_with_saddle=v_donut_with_saddle,
         v_donut_without_saddle=v_donut_without_saddle, v_donut_bad_candidate=v_donut_bad_candidate, v_sample=v_sample)


print('It is the end!')
