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
gamma_indices = np.array([0, 7, 14, 21, 25, 35])
theta_indices =[[3, 6], [10, 13], [17, 20], [24], [28, 31, 34], [38, 41]]
# monostable_pars[gamma_indices] = 1.0 + 0*gamma_indices
L, U, T = HillContpar_to_DSGRN(f, monostable_pars, indices_sources_EMT, indices_targets_EMT)
monostable_pars_test, index_a, index_b = DSGRNpar_to_HillCont(L, T, U)
success = (monostable_pars == monostable_pars_test)
different_index = np.argwhere(monostable_pars_test!=monostable_pars)[:, 0]
print('if gammai is not 1, there is a difference between the two parameter vectors, only in the gammas and thetas elements:')
print(monostable_pars[different_index])
print(monostable_pars_test[different_index])

monostable_pars_rescaled = monostable_pars.copy()
monostable_pars_rescaled[gamma_indices] = 1.0 + 0*gamma_indices
for i in range(len(gamma_indices)):
    gamma_i = monostable_pars[gamma_indices[i]]
    thetas_i = monostable_pars[theta_indices[i]]
    monostable_pars_rescaled[theta_indices[i]] = gamma_i * thetas_i
different_index = np.argwhere(monostable_pars_test != monostable_pars_rescaled)[:, 0]
print(monostable_pars_rescaled[different_index])
print(monostable_pars_test[different_index])


print(' SUCCESS = ', all(monostable_pars_test == monostable_pars_rescaled))
