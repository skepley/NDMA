from hill_model import *
from saddle_finding import *
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
n_sample_side = 10
n_sample = n_sample_side**2
n_second_sample = 10
# a random parameter list
[u, v] = np.meshgrid(np.linspace(0.8, 2.2, n_sample_side), np.linspace(0.8, 2.2, n_sample_side))
u = u.flatten()
u = np.repeat(u, n_second_sample)
v = v.flatten()
v = np.repeat(v, n_second_sample)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample*n_second_sample)])

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
    if SNParameters and SNParameters is not 0:
        for k in range(len(SNParameters)):
            #print('Saddle detected')
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            solutions = np.append(solutions, SNParameters[k])
            if k > 0:
                print('More than one saddle detected!')
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates and badCandidates is not 0:
        print('\nA bad parameter')
        bad_parameters = np.append(bad_parameters, [a_j], axis=0)
        bad_candidates.append(badCandidates)
    printing_statement = 'Completion: ' + str(j) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

    if SNParameters is 0 and badCandidates is 0:
        #print('boooring')
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)

np.savez('averaging_data',
         u=u, v=v, a=a, parameter_full=parameter_full, solutions=solutions, bad_parameters=bad_parameters,
         bad_candidates=bad_candidates, boring_parameters=boring_parameters)
data = np.load('data_center_region_small.npz')

print('\nNumber of bad candidates', len(bad_candidates), 'out of ', n_sample)
print('Number of boring candidates', len(boring_parameters), 'out of ', n_sample)
print('Number of saddles', len(parameter_full), 'out of ', n_sample)

if bad_parameters is not None:
    fig1 = plt.figure()
    dsgrn_plot(bad_parameters, 10)
    plt.title('bad candidates')

if boring_parameters is not None:
    fig1 = plt.figure()
    dsgrn_plot(boring_parameters, 10)
    plt.title('No saddle detected')


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
    if len(index_solution_j)>0:
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
