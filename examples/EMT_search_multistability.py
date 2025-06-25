"""
Search for saddle-node bifurcations in the EMT model

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 11/17/2021
"""

from DSGRN import *
from ndma.DSGRNintegration.DSGRN_tools import DSGRN_parameter_to_NDMA, NDMA_parameter_to_DSGRN
from ndma.bifurcation.saddlenode import SaddleNode
from ndma.basic_models.EMT_model import EMT

gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# define the DSGRN network and pick out a multistable parameter
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
sampler = DSGRN.ParameterSampler(EMT_network)

isFP = lambda morse_node: morse_graph.annotation(morse_node)[0].startswith('FP')
multistable_FP_parameters = []
good_candidate = []

for par_index in range(150):  # parameter_graph_EMT.size()
    parameter = parameter_graph_EMT.parameter(par_index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    morse_nodes = range(morse_graph.poset().size())
    num_stable_FP = sum(1 for node in morse_nodes if isFP(node))
    if num_stable_FP >= 2:
        multistable_FP_parameters.append(par_index)
        break

# check equilibria works correctly. In this region we should find > 1 equilibrium at "d = infty" and fewer as d becomes lower
multistable_region = multistable_FP_parameters[0]  # linear DSGRN index e.g. 127
p = DSGRN_parameter_to_NDMA(EMT_network, multistable_region, edgeCounts)
# inverse action FAILS
region_number = NDMA_parameter_to_DSGRN(EMT_network, f, edgeCounts, np.nan, p)
if region_number is not multistable_region:
    raise NameError('Regions not compatible')

print(len(f.find_equilibria(3, 100, p, uniqueRootDigits=3)))  # finds 3 equilibria
print(len(f.find_equilibria(3, 10, p, uniqueRootDigits=3)))  # finds only 1 equilibrium

# Search for saddle node bifurcation.
# Begin with Equilibria search in bistable region along a line of hill parameters
hill = [2, 10, 20, 30, 50, 100]  # some arbitrary Hill coefficients for a line search
badCandidates = []
saddle_node_found = 0
nEq = []
for d in hill:
    eq = f.find_equilibria(3, d, p, uniqueRootDigits=3)
    print(eq)
    nEq.append(np.shape(eq)[0])
print('Equilibria found: {0}'.format(nEq))
# equilibria counts are [1, 1, 3, 3, 3, 3]


# Saddle node bifurcation search for d in [10, 20]
SNB = SaddleNode(f)
snb = []
for d in [10, 15, 20]:
    snb_d = SNB.find_saddle_node(0, d, p)
    if len(snb_d) > 0:
        snb.append(snb_d[0])
        print('SNB found: {0}'.format(snb_d))

# saddle node found at d ~ 12.53487