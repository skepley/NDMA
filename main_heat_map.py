"""
This code creates the heat map, as presented in the article.
The heat map indicates the value of the Hill coefficient in which a saddle node is taking place depending on the parameter.
It also consider the parameter projection into [0,3]x[0,3] thanks to the DSGRN region definition
"""

from hill_model import *
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import numpy as np
import matplotlib.pyplot as plt
from models.TS_model import ToggleSwitch
import sys


# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
# # # LONG RUN
n_sample_side = 11
n_sample = (n_sample_side*3)**2
n_second_sample = 15*4
# # # SHORT RUN
n_sample_side = 3
n_sample = (n_sample_side*3)**2
n_second_sample = 3*4
# a random parameter list
interpolation_array = np.array([np.linspace(0.85, 1.1, n_sample_side), np.linspace(1.1, 1.9, n_sample_side), np.linspace(1.9, 2.15, n_sample_side)])
[u, v] = np.meshgrid(interpolation_array, interpolation_array)
u = u.flatten()
u = np.repeat(u, n_second_sample)
v = v.flatten()
v = np.repeat(v, n_second_sample)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample * n_second_sample)])
# TODO: decide if we want to keep the fiber sampling or move to dataset creation

n_sample = n_sample * n_second_sample

parameter_full = np.empty(shape=[0, 5])
solutions = np.empty(0)
bad_parameters = np.empty(shape=[0, 5])
bad_candidates = []
boring_parameters = np.empty(shape=[0, 5])
multiple_saddles = np.empty(shape=[0, 5])
for j in range(n_sample):#range(n_sample):
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, [1, 50], a_j)
    if SNParameters and SNParameters != 0:
        for k in range(len(SNParameters)):
            #print('Saddle detected')
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            solutions = np.append(solutions, SNParameters[k])
            if k > 0:
                print('More than one saddle detected!')
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates and badCandidates != 0:
        print('\nA bad parameter')
        bad_parameters = np.append(bad_parameters, [a_j], axis=0)
        bad_candidates.append(badCandidates)
    printing_statement = 'Completion: ' + str(j) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

    if SNParameters == 0 and badCandidates == 0:
        #print('boooring')
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)

np.savez('boundary_averaging_data',
         u=u, v=v, a=a, parameter_full=parameter_full, solutions=solutions, bad_parameters=bad_parameters,
         bad_candidates=bad_candidates, boring_parameters=boring_parameters, n_sample=n_sample)

data = np.load('boundary_averaging_data.npz',allow_pickle=True)


bad_candidates = data.f.bad_candidates
boring_parameters = data.f.boring_parameters
n_sample = data.f.n_sample
parameter_full = data.f.parameter_full
solutions = data.f.solutions



print('\nNumber of bad candidates', len(bad_candidates), 'out of ', n_sample)
print('Number of boring candidates', len(boring_parameters), 'out of ', n_sample)
print('Number of saddles', len(parameter_full), 'out of ', n_sample)

"""if bad_candidates is not None:
    fig1 = plt.figure()
    dsgrn_plot(bad_candidates, 10)
    plt.title('bad candidates')

if boring_parameters is not None:
    fig1 = plt.figure()
    dsgrn_plot(boring_parameters, 10)
    plt.title('No saddle detected')
"""

parameter_DSGRN = parameter_to_DSGRN_coord(parameter_full, 10)
parameter_DSGRN = np.array([parameter_DSGRN[0], parameter_DSGRN[1]])
unique_DSGRN = np.unique(parameter_DSGRN.round(decimals=5), axis=1)
average_sol = np.empty(0)
average_sol_long = 0*solutions
for j in unique_DSGRN.T:
    # work in progress
    index_solution_j = np.where(abs(parameter_DSGRN[0, :] - j[0]) < 5*10**-5)
    index_solution_loc = np.where(abs(parameter_DSGRN[1, :] - j[1]) < 5*10**-5)
    index_solution_j = np.intersect1d(index_solution_j, index_solution_loc)
    if len(index_solution_j) > 0:
        average_sol = np.append(average_sol, np.mean(solutions[index_solution_j]))
        average_sol_long[index_solution_j] = np.mean(solutions[index_solution_j])
    else:
        print('wrong')

fig1 = plt.figure()
dsgrn_heat_plot(parameter_full, average_sol_long, 10)

fig1 = plt.figure()
dsgrn_contour_plot(parameter_full, average_sol_long, 10)

plt.figure()
dsgrn_plot(parameter_full, 10)

print('It is the end!')