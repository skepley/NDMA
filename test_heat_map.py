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
n_sample = 400
# a random parameter list
u = np.random.uniform(2, 2.5, n_sample)
v = np.random.uniform(2, 2.5, n_sample)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample)])

parameter_full = np.empty(shape=[0, 5])
solutions = None
"""for j in range(0):
    print(j)
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, 100, a_j)
    if SNParameters is not 0:
        for k in range(len(SNParameters)):
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            if solutions is None:
                print('There is one saddle node')
                solutions = SNParameters[k]
            else:
                print('There is another saddle node')
                solutions = ezcat(solutions, SNParameters[k])
"""
parameter_full = a
solutions = np.array([3 * np.random.random_sample() for j in range(n_sample)])

plt.scatter(u, v)

dsgrn_plot(parameter_full, 10)

dsgrn_heat_plot(parameter_full, solutions, 10)

print('It is the end!')
