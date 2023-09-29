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


def from_string_to_Hill_data(DSGRN_par_string, domain_size, network, parameter_graph, region):
    D = network.size()
    gamma = np.ones(D)
    L = np.zeros([D, D])
    U = np.zeros([D, D])
    T = np.zeros([D, D])
    indices_domain = []
    indices_input = []
    sample_dict = json.loads(DSGRN_par_string)
    for key, value in sample_dict['Parameter'].items():
        # Get parameter (L, U, or T)
        par_type = key[0]
        # Extract variable names
        node_names = [name.strip() for name in key[2:-1].split('->')]
        node_indices = [network.index(node) for node in node_names]
        if par_type == 'L':
            L[tuple(node_indices)] = value
            indices_domain.append(node_indices[0])
            indices_input.append(node_indices[1])
        elif par_type == 'U':
            U[tuple(node_indices)] = value
        else:  # T
            T[tuple(node_indices)] = value

    delta = U - L
    ell_non_zero = L[np.nonzero(L)]
    theta_non_zero = T[np.nonzero(T)]
    delta_non_zero = delta[np.nonzero(delta)]
    all_pars = np.append(gamma, np.append(ell_non_zero, np.append(theta_non_zero, delta_non_zero)))
    """
    success = (DSGRN.par_index_from_sample(parameter_graph, L, U, T) == region)
    if not success:
        raise ValueError('Debugging error')
    L_new, U_new, T_new = HillContpar_to_DSGRN(all_pars, indices_domain, indices_input, domain_size)
    pars_Hill, index_dom, index_in = DSGRNpar_to_HillCont(L_new, T_new, U_new )
    if np.max(np.abs(pars_Hill - all_pars))>10**-7:
        raise ValueError('Debugging error')
    if DSGRN.par_index_from_sample(parameter_graph, L_new, U_new, T_new) != region:
        raise ValueError('Debugging error')

    L, U, T = HillContpar_to_DSGRN(all_pars, indices_domain, indices_input, domain_size)
    success = (DSGRN.par_index_from_sample(parameter_graph, L, U, T) == region)
    if not success:
        raise ValueError('Debugging error')
    """
    return all_pars, indices_domain, indices_input


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

niter = 5
file_storing = 'EMT_nonlocal_chitest.npz'


def DSGRN_FP_per_index(par_index):
    parameter = parameter_graph_EMT.parameter(par_index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    isFP = lambda morse_node: morse_graph.annotation(morse_node)[0].startswith('FP')
    morse_nodes = range(morse_graph.poset().size())
    num_stable_FP = sum(1 for node in morse_nodes if isFP(node))
    return num_stable_FP


def neighbors_of_given_stability(adjacent_nodes, n_FP):
    num_loc_Nstable = sum(1 for k in adjacent_nodes if DSGRN_FP_per_index(adjacent) == n_FP)
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
    par_index = np.random.randint(0, high=num_parameters)
    num_stable_FP = DSGRN_FP_per_index(par_index)
    if num_stable_FP == 1:
        monostable_FP_parameters.append(par_index)
        adjacent_nodes = parameter_graph_EMT.adjacencies(par_index)
        for adjacent in adjacent_nodes:
            if DSGRN_FP_per_index(adjacent) == 2:
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

    # checking monostability around bistable node
    bistable_adjacent_nodes = parameter_graph_EMT.adjacencies(good_candidate[index][1])
    ratio_bistable = neighbors_of_given_stability(bistable_adjacent_nodes, 2) / len(bistable_adjacent_nodes)

    grade_candidate = np.append(grade_candidate, ratio_monostable ** 2 + ratio_bistable ** 2)

index_param = np.argsort(-grade_candidate)
grade_candidate = grade_candidate[index_param]
good_candidate = np.array(good_candidate)[index_param]

ds = []
dsMinimum = []

correlation_matrix = np.array([[0, 0, 0], [0, 0, 0]])
for n_regions in range(niter):
    for par_index in good_candidate[n_regions]:
        parameternode = parameter_graph_EMT.parameter(par_index)
        par_index = par_index + 1
        p = sampler.sample(parameternode)
        num_stable_FP = DSGRN_FP_per_index(par_index)

        Hill_par, _, _ = from_string_to_Hill_data(p, domain_size_EMT, EMT_network,
                                                  parameter_graph_EMT, parameternode)

        try:
            SNParameters, badCandidates = saddle_node_search(f, [1, 10, 20, 35, 50, 75, 100], Hill_par, ds, dsMinimum,
                                                             maxIteration=100, gridDensity=3, bisectionBool=True)
            if SNParameters == 0:
                n_saddles = 0
            else:
                n_saddles = np.max([len(SNParameters), 2])  # 2 = many
            correlation_matrix[num_stable_FP - 1, n_saddles] += 1

            printing_statement = 'Completion: ' + str(n_regions) + ' out of ' + str(niter) + ', region number ' + str(
                par_index)
            sys.stdout.write('\r' + printing_statement)
            sys.stdout.flush()
        except Exception as error:
            # turn an error into a warning and print the associated tag
            warnings.warn(str("An exception occurred:" + type(error).__name__ + "–" + str(error)))

try:
    data = np.load(file_storing, allow_pickle=True)
    old_niter = data.f.niter
except:
    old_niter = 0

if old_niter < niter:
    np.savez(file_storing, correlation_matrix=correlation_matrix, niter=niter)

print('Correlation matrix\n')
print(correlation_matrix)


def print_pvalue_comment(p):
    if p <= 0.05:
        print('We reject the null hypothesis: there is correlation between DSGRN and numerical saddles\n')
    else:
        print(
            'We cannot reject the null hypothesis: there is NO proven correlation between DSGRN and numerical saddles\n')
    print('p-value = ', p)


try:
    _, p, _, _ = chi2_contingency(correlation_matrix)
    print('Results with multiple saddles:\n')
    print_pvalue_comment(p)
except:
    print('chi test failed')

try:
    _, p, _, _ = chi2_contingency(correlation_matrix[:, 0:1])
    print('\nResults without multiple saddles:\n')
    print_pvalue_comment(p)
except:
    print('chi test failed')

print('It is the end!')
