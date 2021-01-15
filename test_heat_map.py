from hill_model import *
from saddle_finding import *
from toggle_switch_heat_functionalities import *

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
n_sample = 40
# a random parameter list
u = np.array([1 + np.random.random_sample() for j in range(n_sample)])
v = np.array([1 + np.random.random_sample() for j in range(n_sample)])
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample)])

parameter_full = np.empty(shape=[0, 5])
solutions = None
for j in range(n_sample):
    print(j)
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, 100, a_j)
    if SNParameters is not 0:
        for k in range(len(SNParameters)):
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            if solutions is None:
                print('There is a saddle node')
                solutions = SNParameters[k]
            else:
                solutions = ezcat(solutions, SNParameters[k])

dsgrn_heat_plot(parameter_full, solutions)
print('It is the end!')
