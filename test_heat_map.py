from hill_model import *
from saddle_finding import *
from toggle_switch_heat_functionalities import *
import matplotlib.pyplot as plt

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
n_sample = 10000
# a random parameter list
u = 1 + np.random.uniform(-0.1, 1.1, n_sample)
v = 1 + np.random.uniform(-0.1, 1.1, n_sample)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample)])

parameter_full = np.empty(shape=[0, 5])
solutions = None
bad_parameters = np.empty(shape=[0, 5])
boring_parameters = np.empty(shape=[0, 5])
multiple_saddles = np.empty(shape=[0, 5])
for j in range(n_sample):
    print(j)
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, 100, a_j)
    if SNParameters is not 0:
        for k in range(len(SNParameters)):
            print('Saddle detected')
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            if solutions is None:
                solutions = SNParameters[k]
            else:
                solutions = ezcat(solutions, SNParameters[k])
            if k > 0:
                print('More than one saddle detected!')
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates is not 0:
        for k in range(len(badCandidates)):
            print('A bad parameter')
            bad_parameters = np.append(bad_parameters, [a_j], axis=0)
    if SNParameters is 0 and badCandidates is 0:
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)


np.savez('data_center_region', u=u, v=v, a=a, parameter_full=parameter_full, solutions=solutions, bad_parameters=bad_parameters, boring_parameters=boring_parameters)
np.load('data_center_region.npz')

if bad_parameters is not None:
    dsgrn_plot(bad_parameters, 10)
    plt.title('bad candidates')

if boring_parameters is not None:
    dsgrn_plot(boring_parameters, 10)
    plt.title('No saddle detected')

dsgrn_heat_plot(parameter_full, solutions, 10)


print('It is the end!')
