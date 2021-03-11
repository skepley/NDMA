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

interestingIndex = 0
# this index indicates what we parameter we are changing to find the saddle node

# size of the sample
n_sample = 10
# a random parameter list
u = np.linspace(0.1, 3, n_sample)#1 + np.random.uniform(-0.1, 1.1, n_sample)
v = np.full(n_sample, 1.5) #1 + np.random.uniform(-0.1, 1.1, n_sample)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample)])

hill_range = np.linspace(4, 15, 5)

parameter_full = np.empty(shape=[0, 5])
solutions = None
bad_parameters = np.empty(shape=[0, 5])
bad_candidates = []
boring_parameters = np.empty(shape=[0, 5])
multiple_saddles = np.empty(shape=[0, 5])


for hill in hill_range:
    for j in range(n_sample):
        # print(j)
        a_j = a[j, :]
        a_j = ezcat(hill, a_j[:interestingIndex], a_j[interestingIndex+1:]) # a_j has hill and skips the interestingIndex
        SNParameters, badCandidates = find_saddle_coef(f, [0, a_j[interestingIndex]], a_j, interestingIndex) # totally arbitrary starting point
        if SNParameters and SNParameters is not 0:
            for k in range(len(SNParameters)):
                #print('Saddle detected')
                parameter_full = np.append(parameter_full, [a_j], axis=0)
                if solutions is None:
                    solutions = SNParameters[k]
                else:
                    solutions = ezcat(solutions, SNParameters[k])
                if k > 0:
                    # print('More than one saddle detected!')
                    multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
        if badCandidates and badCandidates is not 0:
            # print('A bad parameter')
            bad_parameters = np.append(bad_parameters, [a_j], axis=0)
            bad_candidates.append(badCandidates)

        if SNParameters is 0 and badCandidates is 0:
            boring_parameters = np.append(boring_parameters, [a_j], axis=0)
        # Check for saddles in the other direction

        SNParameters, badCandidates = find_saddle_coef(f, [a_j[interestingIndex], 100], a_j, interestingIndex)
        if SNParameters and SNParameters is not 0:
            for k in range(len(SNParameters)):
                print('\nSaddle detected\n')
                parameter_full = np.append(parameter_full, [a_j], axis=0)
                if solutions is None:
                    solutions = SNParameters[k]
                else:
                    solutions = ezcat(solutions, SNParameters[k])
                if k > 0:
                    # print('More than one saddle detected!')
                    multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
        if badCandidates and badCandidates is not 0:
            print('\nA bad parameter\n')
            bad_parameters = np.append(bad_parameters, [a_j], axis=0)
            bad_candidates.append(badCandidates)

        if SNParameters is 0 and badCandidates is 0:
            boring_parameters = np.append(boring_parameters, [a_j], axis=0)

        printing_statement = 'Completion: ' + str(np.where(hill_range == hill)[0] * n_sample + j) + ' out of ' + str(len(hill_range) * n_sample)
        sys.stdout.write('\r' + printing_statement)
        sys.stdout.flush()

"""
np.savez('data_center_region_small',
         u=u, v=v, a=a, parameter_full=parameter_full, solutions=solutions, bad_parameters=bad_parameters,
         bad_candidates=bad_candidates, boring_parameters=boring_parameters)
np.load('data_center_region_small.npz')
"""

if np.shape(bad_parameters)[0] is not 0:
    fig1 = plt.figure()
    dsgrn_plot(bad_parameters, 10)
    plt.title('bad candidates')

if np.shape(boring_parameters)[0] is not 0:
    print('Some other time')
#    fig1 = plt.figure()
#    dsgrn_plot(boring_parameters, 10)
#    plt.title('No saddle detected')

if np.shape(parameter_full)[0] > 3:
    fig1 = plt.figure()
    dsgrn_heat_plot(parameter_full, solutions, 10)

print('It is the end!')
