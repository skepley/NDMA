"""
Search for saddle-node bifurcations in the EMT model

    Author: Elena Queirolo
    Email: elena.queirolo@tum.de
    Created: 12/09/2023
"""
import warnings

from models.EMT_model import *
from saddle_finding_functionalities import *
from create_dataset import *
import sys
from scipy.stats import chi2_contingency
import DSGRN
import json
from DSGRN_functionalities import from_string_to_Hill_data, DSGRN_FP_per_index, random_in_region


# set EMT-specific elements
gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# create network from file
EMT_network = DSGRN.Network("EMT.txt")
# graph_EMT = graphviz.Source(EMT_network.graphviz())
# graph_EMT.view()
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
# building the sampler (later used to create a sample parameter per each selected region)
sampler = DSGRN.ParameterSampler(EMT_network)
num_parameters = parameter_graph_EMT.size()
domain_size_EMT = 6

# print('test: is domain size equal to network size? domain size =', domain_size_EMT, 'network size = ', EMT_network.size())

niter = 4
file_storing = 'EMT_nonlocal_chitest_morepoints_small.npz'


def neighbors_of_given_stability(adjacent_nodes, n_FP):
    num_loc_Nstable = sum(1 for adjacent in adjacent_nodes if DSGRN_FP_per_index(adjacent, parameter_graph_EMT) == n_FP)
    # num_loc_Nstable = 0
    # for adjacent in adjacent_nodes:
    #    if DSGRN_FP_per_index(adjacent) == n_FP:
    #        num_loc_Nstable = num_loc_Nstable + 1
    return num_loc_Nstable


'''
Here, we select the parameter regions we will want to research
'''

monostable_FP_parameters = []
bistable_FP_parameters = []
multistable_FP_parameters = []
good_candidate = []
n_tested_regions = -1
while len(good_candidate) < niter * 3:
    n_tested_regions = n_tested_regions + 1  # for statistics reasons
    par_index = np.random.randint(0, high=1000)# high=num_parameters)
    num_stable_FP = DSGRN_FP_per_index(par_index, parameter_graph_EMT)
    if num_stable_FP == 1:
        monostable_FP_parameters.append(par_index)
        adjacent_nodes = parameter_graph_EMT.adjacencies(par_index)
        for adjacent in adjacent_nodes:
            if DSGRN_FP_per_index(adjacent, parameter_graph_EMT) == 2:
                good_candidate.append((par_index, adjacent))
    elif num_stable_FP == 2:
        bistable_FP_parameters.append(par_index)
    elif num_stable_FP > 2:
        multistable_FP_parameters.append(par_index)

num_monostable_params = len(monostable_FP_parameters)
num_bistable_params = len(bistable_FP_parameters)
num_multistable_params = len(multistable_FP_parameters)
num_candidates = len(good_candidate)

print('Number of parameters in the parameter graph: ' + str(num_parameters))
print('Monostable parameters in the parameter graph: ' + str(num_monostable_params))
print('Bistable parameters in the parameter graph: ' + str(num_bistable_params))
print('Multistable parameters in the parameter graph: ' + str(num_multistable_params))
print('Good parameters in the parameter graph: ' + str(num_candidates))
print('Tested parameters: ' + str(n_tested_regions))

# refine the search: to each good candidate count the number of monostable adjacent nodes / number of adjacent nodes and
# the same for the bistable node: we want as many monostable nodes close to the monostable node and the bistable node
grade_candidate = np.array([])
for index in range(num_candidates):
    # checking monostability around monostable node
    monostable_adjacent_nodes = parameter_graph_EMT.adjacencies(good_candidate[index][0])
    ratio_monostable = neighbors_of_given_stability(monostable_adjacent_nodes, 1) / len(monostable_adjacent_nodes)

    # checking bistability around bistable node
    bistable_adjacent_nodes = parameter_graph_EMT.adjacencies(good_candidate[index][1])
    ratio_bistable = neighbors_of_given_stability(bistable_adjacent_nodes, 1) / len(bistable_adjacent_nodes)

    grade_candidate = np.append(grade_candidate, ratio_monostable ** 2 + ratio_bistable ** 2)

index_param = np.argsort(-grade_candidate)
grade_candidate = grade_candidate[index_param]
good_candidate = np.array(good_candidate)[index_param, :]

ds = []
dsMinimum = []

correlation_matrix = np.array([[0, 0, 0], [0, 0, 0]])
print('\nstarting saddle node computations \n\n')
for n_regions in range(niter):
    # for par_index in good_candidate[n_regions][1]:
    for i in range(2):
        par_index = good_candidate[n_regions][i]
        parameternode = parameter_graph_EMT.parameter(par_index)
        # par_index = par_index + 1
        p = sampler.sample(parameternode)
        num_stable_FP = DSGRN_FP_per_index(par_index, parameter_graph_EMT)
        print('Number of fixed points: ', num_stable_FP)

        # Hill_par, _, _ = from_string_to_Hill_data(p, EMT_network)

        point_cloud = random_in_region(par_index, EMT_network, parameter_graph_EMT, 1)

        for Hill_par in point_cloud:
            gridDensity = 2
            #nEq1, _ = count_equilibria(f, 1, Hill_par, gridDensity)
            nEq100, _ = count_equilibria(f, 100, Hill_par, gridDensity)
            #print('n. eqs at hill coef = 1 is ', nEq1)
            print('n. eqs at hill coef = 100 is ', nEq100)
            #continue
            #for Hill_par in point_cloud:
            #SNParameters, otherBif = saddle_node_search(f, [1, 10, 20, 50, 100], Hill_par, ds, dsMinimum,
            #                                            maxIteration=100, gridDensity=3, bisectionBool=True)
            try:
                if nEq100 == 1:#SNParameters == 0:
                    n_saddles_idx = 0
                else:
                    # n_saddles_idx = np.max([len(SNParameters)-1, 2])  # more than 0 = many
                    n_saddles_idx = 1
                if n_saddles_idx > 0:
                    print('There are ', nEq100, ' equilibria \n')
                correlation_matrix[num_stable_FP - 1, n_saddles_idx] += 1

            except Exception as error:
                # turn an error into a warning and print the associated tag
                warnings.warn(str("An exception occurred:" + type(error).__name__ + "–" + str(error)))
    printing_statement = 'Completion: ' + str(n_regions + 1) + ' out of ' + str(niter) + ', region number ' + str(
        good_candidate[n_regions]) + '\n'
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()
# sys.exit()

try:
    data = np.load(file_storing, allow_pickle=True)
    old_niter = data.f.niter
except:
    old_niter = 0

if old_niter < niter:
    np.savez(file_storing, correlation_matrix=correlation_matrix, niter=niter)

print('\n\nCorrelation matrix\n')
print(correlation_matrix)
print('Rows: number of DSGRN fixed points, Columns: number of found equilibria or saddle node')

def print_pvalue_comment(p):
    if p <= 0.05:
        print('We reject the null hypothesis: there is correlation between DSGRN and numerical saddles\n')
    else:
        print(
            'We cannot reject the null hypothesis: there is NO proven correlation between DSGRN and numerical saddles\n')
    print('p-value = ', p)


try:
    print('Results with multiple saddles:\n')
    _, p, _, _ = chi2_contingency(correlation_matrix)
    print_pvalue_comment(p)
except Exception as error:
    # turn an error into a warning and print the associated tag
    warnings.warn(str("An exception occurred:" + type(error).__name__ + "–" + str(error)))
    print('Chi2 test failed')

try:
    print('\nResults without multiple saddles:\n')
    _, p, _, _ = chi2_contingency(correlation_matrix[:, 0:1])
    print_pvalue_comment(p)
except Exception as error:
    # turn an error into a warning and print the associated tag
    warnings.warn(str("An exception occurred:" + type(error).__name__ + "–" + str(error)))
    print('Chi2 test failed')

print('It is the end!')
