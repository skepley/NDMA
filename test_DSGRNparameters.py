import numpy as np
import scipy
import matplotlib.pyplot as plt
import graphviz
import json
from DSGRN_functionalities import *
from models.EMT_model import *


# create network from file
EMT_network = DSGRN.Network("EMT.txt")
# graph_EMT = graphviz.Source(EMT_network.graphviz())
# graph_EMT.view()
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)

# look into a parameter region
monostable_region = 44
bistable_region = 164

# sampling from each region
sampler = DSGRN.ParameterSampler(EMT_network)

monostable_parameternode = parameter_graph_EMT.parameter(monostable_region)
monostable_parameter = sampler.sample(monostable_parameternode)

bistable_parameternode = parameter_graph_EMT.parameter(bistable_region)
bistable_parameter = sampler.sample(bistable_parameternode)

# extract sheer data??
domain_size_EMT = 6
bistable_pars, _, _ = from_string_to_Hill_data(bistable_parameter, EMT_network)
monostable_pars, indices_sources_EMT, indices_targets_EMT = from_string_to_Hill_data(monostable_parameter, EMT_network)


# define the EMT hill model
gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

monostable_pars = np.array(range(42)) + 1.0
gamma_indices = np.array([7, 14, 21, 25, 35])
monostable_pars[gamma_indices] = 1.0 + 0*gamma_indices
L, U, T = HillContpar_to_DSGRN(f, monostable_pars, indices_sources_EMT, indices_targets_EMT)
monostable_pars_test, index_a, index_b = DSGRNpar_to_HillCont(L, T, U)
success = (monostable_pars == monostable_pars_test)
print(success)
