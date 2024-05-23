import numpy as np
import scipy
import matplotlib.pyplot as plt
import graphviz
import json
from DSGRN_functionalities import *
from models.EMT_model import *

# # # # # FIRST TEST: parameters get back the same
# create network from file
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)

# sampling from each region
sampler = DSGRN.ParameterSampler(EMT_network)

random_region = 77

parameternode = parameter_graph_EMT.parameter(random_region)
DSGRNparameter = sampler.sample(parameternode)

# extract sheer data??
_, indices_sources_EMT, indices_targets_EMT = from_string_to_Hill_data(DSGRNparameter, EMT_network)

# define the EMT hill model
f = def_emt_hill_model()
n_parameters = 42

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


# # # # # # # # SECOND TEST: assign correct regions
# look into a parameter region
monostable_region = 8 # 44
bistable_region = 128 # 164

monostable_parameternode = parameter_graph_EMT.parameter(monostable_region)
monostable_parameter = sampler.sample(monostable_parameternode)

bistable_parameternode = parameter_graph_EMT.parameter(bistable_region)
bistable_parameter = sampler.sample(bistable_parameternode)

# extract sheer data??
domain_size_EMT = 6

L, U, T = from_string_to_DSGRN_data(bistable_parameter, EMT_network)
extended_region_number = DSGRN.par_index_from_sample(parameter_graph_EMT, L, U, T)
print('test 2.1: is the DSGRN region correctly computed if we skip passing by Hill?',
      bistable_region == extended_region_number)

bistable_pars, indices_sources_EMT, indices_targets_EMT = DSGRNpar_to_HillCont(L, U, T)

L_test, U_test, T_test = HillContpar_to_DSGRN(f, bistable_pars, indices_sources_EMT, indices_targets_EMT)

print('Test 2.2: does transforming to Hill and back change the parameter? (3 parts)')
print('L:', (L == L_test).all())
print('T:', (T == T_test).all())
print('U:', (U == U_test).all())

print('Test 2.3: regions maintained when switching to Hill pars? (2 parts)')
return_region_number = DSGRN.par_index_from_sample(parameter_graph_EMT, L, U, T)

L, U, T = HillContpar_to_DSGRN(f, bistable_pars, indices_sources_EMT,
                                                  indices_targets_EMT)
return_region_number = DSGRN.par_index_from_sample(parameter_graph_EMT, L, U, T)
print(return_region_number == bistable_region)
test23 = (bistable_region == global_par_to_region(f, bistable_pars, parameter_graph_EMT, indices_sources_EMT,
                                                  indices_targets_EMT))
print(test23)

print('Test 2.4: regions correctly picked from a list?')
test_bistable_region = par_to_region(f, bistable_pars, np.array([8, 128]), parameter_graph_EMT, indices_sources_EMT,
                                                  indices_targets_EMT)
print(test_bistable_region == 1)

monostable_parameternode = parameter_graph_EMT.parameter(monostable_region)
monostable_DSGRNparameter = sampler.sample(monostable_parameternode)

# extract sheer data??
monostable_pars, indices_sources_EMT, indices_targets_EMT = from_string_to_Hill_data(monostable_DSGRNparameter, EMT_network)
test_monostable_region = par_to_region(f, monostable_pars, np.array([8, 128]), parameter_graph_EMT, indices_sources_EMT,
                                                  indices_targets_EMT)
print(test_monostable_region == 0)

assign_region = par_to_region_wrapper(f, np.array([8, 128]), parameter_graph_EMT, indices_sources_EMT, indices_targets_EMT)
print('bistable and monostable regions', assign_region(np.array([bistable_pars, monostable_pars]).T) == [1, 0])