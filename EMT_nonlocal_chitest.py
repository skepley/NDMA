"""
Search for saddle-node bifurcations in the EMT model

    Author: Elena Queirolo
    Email: elena.queirolo@tum.de
    Created: 12/09/2023
"""
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

gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

niter = 500
file_storing = 'EMT_nonlocal_chitest.npz'

# create network from file
EMT_network = DSGRN.Network("EMT.txt")
# graph_EMT = graphviz.Source(EMT_network.graphviz())
# graph_EMT.view()
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)

# look into a parameter region
parameterindex = 64
special_parameternode = parameter_graph_EMT.parameter(parameterindex)
# print(special_parameternode.inequalities())

# sampling a special parameter node
sampler = DSGRN.ParameterSampler(EMT_network)
sampler.sample(special_parameternode)

isFP = lambda morse_node: morse_graph.annotation(morse_node)[0].startswith('FP')

ds = []
dsMinimum = []

mono_nosaddle = 0
mono_saddle = 0
mono_manysaddles = 0
bi_nosaddle = 0
bi_saddle = 0
bi_manysaddles = 0
for par_index in range(niter):  # parameter_graph_EMT.size()
    parameternode = parameter_graph_EMT.parameter(par_index)
    domain_graph = DSGRN.DomainGraph(parameternode)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    morse_nodes = range(morse_graph.poset().size())
    num_stable_FP = sum(1 for node in morse_nodes if isFP(node))
    if num_stable_FP > 2:
        continue
    p = sampler.sample(parameternode)
    domain_size_EMT = 6
    Hill_par, _, _ = from_string_to_Hill_data(p, domain_size_EMT, EMT_network,
                                                   parameter_graph_EMT, parameternode)
    SNParameters, badCandidates = saddle_node_search(f, [1, 10, 20, 35, 50, 75, 100], Hill_par, ds, dsMinimum,
                                                     maxIteration=100, gridDensity=3, bisectionBool=True)
    if num_stable_FP == 1:
        if (SNParameters) == 0:
            mono_nosaddle = mono_nosaddle + 1
        elif (SNParameters) == 1:
            mono_saddle = mono_saddle + 1
        else:
            mono_manysaddles = mono_manysaddles + 1
    else:
        if (SNParameters) == 0:
            bi_nosaddle = bi_nosaddle + 1
        elif (SNParameters) == 1:
            bi_saddle = bi_saddle + 1
        else:
            bi_manysaddles = bi_manysaddles + 1
    printing_statement = 'Completion: ' + str(par_index + 1) + ' out of ' + str(niter)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

try:
    data = np.load(file_storing, allow_pickle=True)
    old_niter = data.f.niter
except:
    old_niter = 0

if old_niter < niter:
    np.savez(file_storing, mono_nosaddle=mono_nosaddle,
             mono_saddle=mono_saddle,
             mono_manysaddles=mono_manysaddles,
             bi_nosaddle=bi_nosaddle,
             bi_saddle=bi_saddle,
             bi_manysaddles=bi_manysaddles, niter=niter)

mat_for_chi_test_LARGE = np.array(
    [[mono_nosaddle, mono_saddle,  mono_manysaddles],
     [bi_nosaddle, bi_saddle, bi_manysaddles]])
mat_for_chi_test = np.array(
    [[mono_nosaddle, mono_saddle],
     [bi_nosaddle, bi_saddle]])
print('Correlation matrix\n')
print(mat_for_chi_test)

unused, p, a, b = chi2_contingency(mat_for_chi_test_LARGE)
print('Results with multiple saddles:\n')
if p <= 0.05:
    print('We reject the null hypothesis: there is correlation between DSGRN and numerical saddles\n')
else:
    print('We cannot reject the null hypothesis: there is NO proven correlation between DSGRN and numerical saddles\n')

unused, p, a, b = chi2_contingency(mat_for_chi_test)
print('\nResults without multiple saddles:\n')
if p <= 0.05:
    print('We reject the null hypothesis: there is correlation between DSGRN and numerical saddles\n')
else:
    print('We cannot reject the null hypothesis: there is NO proven correlation between DSGRN and numerical saddles\n')

print('p-value = ', p)
print('It is the end!')
