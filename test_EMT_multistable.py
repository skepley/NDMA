"""
Search for saddle-node bifurcations in the EMT model

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 11/17/2021
"""
from models.EMT_model import *
from DSGRN import *
from DSGRN_tools import *
import re

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

multistable_region = multistable_FP_parameters[0]  # linear DSGRN index e.g. 127
p = DSGRN_parameter_to_NDMA(EMT_network, multistable_region, edgeCounts)
print(len(f.find_equilibria(3, 100, p, uniqueRootDigits=3)))
print(len(f.find_equilibria(3, 10, p, uniqueRootDigits=3)))




