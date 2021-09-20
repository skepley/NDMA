from hill_model import *
from models.TS_model import ToggleSwitch
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import sys
from os.path import isfile
from create_dataset import *
from scipy.stats import chi2_contingency

file_storing = 'chi_test_data.npz'

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
n_sample = 10 ** 2
file_name = 'TS_data_100000.npz'
try:
    np.load(file_name)
except FileNotFoundError:
    n = 100000
    TS_region(100000, file_name)

file_storing = 'chi_test_data_100000.npz'

data_subsample, region_subsample, coefs = subsample(file_name, n_sample)
a = data_subsample

n_center_region = 0
n_center_with_saddle = 0
wrong_parity_center = 0
n_center_without_saddle = 0
n_center_bad_candidate = 0
n_donut = 0
n_donut_with_saddle = 0
wrong_parity_donut = 0
n_donut_without_saddle = 0
n_donut_bad_candidate = 0

for j in range(n_sample):  # range(n_sample):
    a_j = a[:, j]
    region_j = region_subsample[j]
    if region_j == 5:
        n_center_region = n_center_region + 1
    else:
        n_donut = n_donut + 1
    SNParameters, badCandidates = find_saddle_coef(f, [1, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500], a_j)
    if SNParameters and SNParameters != 0:
        if region_j == 4:
            if len(SNParameters) == 1 or len(SNParameters) == 3:
                n_center_with_saddle = n_center_with_saddle + 1
            else:
                wrong_parity_center = wrong_parity_center + 1
        else:
            if len(SNParameters) == 2:
                n_donut_with_saddle = n_donut_with_saddle + 1
            else:
                wrong_parity_donut = wrong_parity_donut + 1
    elif badCandidates and badCandidates != 0:
        if region_j == 4:
            n_center_bad_candidate = n_center_bad_candidate + 1
        else:
            n_donut_bad_candidate = n_donut_bad_candidate + 1
    else:
        if region_j == 4:
            n_center_without_saddle = n_center_without_saddle + 1
        else:
            n_donut_without_saddle = n_donut_without_saddle + 1
    printing_statement = 'Completion: ' + str(j+1) + ' out of ' + str(n_sample)
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
    v_wrong_parity_center = data.f.v_wrong_parity_center
    v_donut = data.f.v_donut
    v_donut_with_saddle = data.f.v_donut_with_saddle
    v_donut_without_saddle = data.f.v_donut_without_saddle
    v_donut_bad_candidate = data.f.v_donut_bad_candidate
    v_sample = data.f.v_sample
    v_wrong_parity_donut = data.f.v_wrong_parity_donut

    v_center_region = np.append(v_center_region, np.array([n_center_region]))
    v_center_with_saddle = np.append(v_center_with_saddle, np.array([n_center_with_saddle]))
    v_wrong_parity_center = np.append(v_wrong_parity_center, np.array([wrong_parity_center]))
    v_center_without_saddle = np.append(v_center_without_saddle, np.array([n_center_without_saddle]))
    v_center_bad_candidate = np.append(v_center_bad_candidate, np.array([n_center_bad_candidate]))
    v_donut = np.append(v_donut, np.array([n_donut]))
    v_donut_with_saddle = np.append(v_donut_with_saddle, np.array([n_donut_with_saddle]))
    v_donut_without_saddle = np.append(v_donut_without_saddle, np.array([n_donut_without_saddle]))
    v_donut_bad_candidate = np.append(v_donut_bad_candidate, np.array([n_donut_bad_candidate]))
    v_wrong_parity_donut = np.append(v_wrong_parity_donut, np.array([wrong_parity_donut]))
    v_sample = np.append(v_sample, np.array([n_sample]))

except:
    v_center_region = np.array([n_center_region])
    v_center_with_saddle = np.array([n_center_with_saddle])
    v_wrong_parity_center = np.array([wrong_parity_center])
    v_center_without_saddle = np.array([n_center_without_saddle])
    v_center_bad_candidate = np.array([n_center_bad_candidate])
    v_donut = np.array([n_donut])
    v_donut_with_saddle = np.array([n_donut_with_saddle])
    v_wrong_parity_donut = np.array([wrong_parity_donut])
    v_donut_without_saddle = np.array([n_donut_without_saddle])
    v_donut_bad_candidate = np.array([n_donut_bad_candidate])
    v_sample = np.array([n_sample])

np.savez(file_storing,
         v_center_region=v_center_region, v_center_with_saddle=v_center_with_saddle,
         v_center_without_saddle=v_center_without_saddle,
         v_center_bad_candidate=v_center_bad_candidate, v_donut=v_donut, v_donut_with_saddle=v_donut_with_saddle,
         v_donut_without_saddle=v_donut_without_saddle, v_donut_bad_candidate=v_donut_bad_candidate, v_sample=v_sample)

data = np.load(file_storing)

mat_for_chi_test = [[np.sum(v_center_with_saddle), np.sum(v_center_without_saddle), np.sum(v_center_bad_candidate), np.sum(v_wrong_parity_center)], [np.sum(v_donut_with_saddle), np.sum(v_donut_without_saddle), np.sum(v_donut_bad_candidate),np.sum(v_wrong_parity_donut)]]

unused, p = chi2_contingency(mat_for_chi_test)

if p <= 0.05:
    print('We reject the null hypothesis: there is correlation between saddles and center region')
else:
    print('We cannot reject the null hypothesis: there is NO proven correlation between saddles and center region')

print('It is the end!')
