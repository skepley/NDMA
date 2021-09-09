"""
Fixing a slice of parameters, we see the behaviour of the saddle nodes when changing a single parameter (gamma) with
respect to which we have hysteresis
"""

from models.TS_model import ToggleSwitch
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import matplotlib.pyplot as plt
import sys

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
n_sample = 51
n_second_sample = 25
# a random parameter list
u_short = np.linspace(0.5, 2.5, n_sample)#1 + np.random.uniform(-0.1, 1.1, n_sample)
u = np.repeat(u_short, n_second_sample)
v = np.full(n_sample*n_second_sample, 1.5)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample*n_second_sample)])

parameter_full = np.empty(shape=[0, 5])
solutions = None
bad_parameters = np.empty(shape=[0, 5])
bad_candidates = []
bad_candidates_y = []
boring_parameters = np.empty(shape=[0, 5])
boring_y = []
multiple_saddles = np.empty(shape=[0, 5])
u_info_n_vs_gamma = np.empty(0)
n_info_n_vs_gamma = np.empty(0)

for j in range(n_sample * n_second_sample):
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, [1, 50], a_j)
    if SNParameters and SNParameters is not 0:
        for k in range(len(SNParameters)):
            #print('Saddle detected')

            u_info_n_vs_gamma = np.append(u_info_n_vs_gamma, u[j])
            n_info_n_vs_gamma = np.append(n_info_n_vs_gamma, SNParameters[k])
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            if solutions is None:
                solutions = SNParameters[k]
            else:
                solutions = ezcat(solutions, SNParameters[k])
            if k > 0:
                print('More than one saddle detected!')
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates and badCandidates is not 0:
        # print('\nA bad parameter')
        bad_parameters = np.append(bad_parameters, [a_j], axis=0)
        bad_candidates.append(badCandidates)
        bad_candidates_y.append(u[np.mod(j, n_sample)])
    printing_statement = 'Completion: ' + str(j+1) + ' out of ' + str(n_sample * n_second_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

    if SNParameters is 0 and badCandidates is 0:
        #print('boooring')
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)
        boring_y.append(u[np.mod(j, n_sample)])
print('\n')

average_hill = np.empty(n_sample)
for i in range(n_sample):
    average_hill[i] = np.mean(n_info_n_vs_gamma[np.where(u_info_n_vs_gamma == u_short[i])])


count_hill = np.empty(n_sample)
number_saddles = []
for i in range(n_sample):
    count_hill[i] = np.shape(n_info_n_vs_gamma[np.where(u_info_n_vs_gamma == u_short[i])])[0]
    #number_saddles.append(len(count_hill[i]))

np.savez('data_n_wrt_gamma',
         u=u, v=v, a=a, u_info_n_vs_gamma=u_info_n_vs_gamma, n_info_n_vs_gamma=n_info_n_vs_gamma, average_hill=average_hill,
         count_hill=count_hill)
data = np.load('data_n_wrt_gamma.npz')

plt.figure()
plt.title('Hill coef VS x')
plt.plot(data.f.u_info_n_vs_gamma, data.f.n_info_n_vs_gamma, 'ro')


plt.figure()
plt.plot(np.unique(data.f.u), data.f.average_hill, 'o')
plt.title('Average Hill coefficient')

plt.figure()
plt.plot(np.unique(data.f.u), data.f.count_hill, 'o')
plt.title('Number of saddles out of sample number ' + str(n_second_sample))


#print('\nNumber of bad candidates', len(bad_candidates), 'out of ', n_sample)
#print('Number of boring candidates', len(boring_parameters), 'out of ', n_sample)
#print('Number of saddles', len(parameter_full), 'out of ', n_sample)

"""
if bad_parameters is not None:
    fig1 = plt.figure()
    dsgrn_plot(bad_parameters, 10)
    plt.title('bad candidates')

if boring_parameters is not None:
    fig1 = plt.figure()
    dsgrn_plot(boring_parameters, 10)
    plt.title('No saddle detected')
fig1 = plt.figure()
dsgrn_heat_plot(parameter_full, solutions, 10)
"""
print('It is the end!')
