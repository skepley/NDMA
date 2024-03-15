"""
Search for saddle-node bifurcations in the EMT model

    Author: Elena Queirolo
    Email: elena.queirolo@tum.de
    Created: 12/09/2023
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from models.EMT_model import *
from saddle_finding_functionalities import *
from create_dataset import *
import sys
from scipy.stats import chi2_contingency
import DSGRN
import json
from DSGRN_functionalities import *

# reminder:
# return_region_number = DSGRN.par_index_from_sample(parameter_graph, L, U, T)

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

# this is just to get the order of the parameters
random_region = 1
a_parameternode = parameter_graph_EMT.parameter(1)
a_parameter = sampler.sample(a_parameternode)
a_pars, indices_domain_EMT, indices_input_EMT = from_string_to_Hill_data(a_parameter, EMT_network)

# len(indices_domain_EMT) = 12

size_L = 12
size_sample = 1000
mu, sigma = .5, 1. # mean and standard deviation
# good to get bistability : .5, 1. with lognormal distribution and means scaled 1-10-100
# best to get bistability : .5, .1 with lognormal distribution and means scaled 1-4-8
L_sample = np.abs(np.random.lognormal(mu, sigma, [size_sample, size_L]))
U_sample = np.abs(np.random.lognormal(mu, sigma, [size_sample, size_L]))*10
T_sample = np.abs(np.random.lognormal(mu, sigma, [size_sample, size_L]))*100
gamma = np.ones([size_sample, 6])
global_sample = np.concatenate((gamma, L_sample, U_sample, T_sample), axis=1)
# print(global_sample[1])

parameter_graph_EMT.parameter(1)

FP_random_pars = np.array([par_to_n_eqs(global_sample[i], parameter_graph_EMT, indices_domain_EMT, indices_input_EMT, domain_size_EMT) for i in range(size_sample)])
# print(FP_random_pars)
regions = np.array([global_par_to_region(global_sample[i], parameter_graph_EMT, indices_domain_EMT, indices_input_EMT, domain_size_EMT) for i in range(size_sample)])

n_Monostable = sum(FP_random_pars==1)

n_Bistable = sum(FP_random_pars==2)

print('Number monostable parameters = ', n_Monostable)
print('Number bistable parameters = ', n_Bistable)
print('Regions numbers sample = ', regions[:10])

# sys.exit()

ds = []
dsMinimum = []

correlation_matrix = np.array([[0, 0, 0], [0, 0, 0]])
print('\nstarting saddle node computations \n\n')
for Hill_par in global_sample:
    #for par_index in good_candidate[n_regions][1]:
    num_stable_FP = par_to_n_eqs(Hill_par, parameter_graph_EMT, indices_domain_EMT, indices_input_EMT, domain_size_EMT)
    if num_stable_FP != 2:
        print('parameter monostable - skipped')
        continue
    gridDensity = 3
    try:
        hill_coef = 2
        nEq1, _ = count_equilibria(f, hill_coef, Hill_par, gridDensity)
        print('at Hill coef', hill_coef, 'n. eqs is ', nEq1)
        # print('Testing at hill coef = ', hill_coef)
        #nEq1, b = count_equilibria(f, hill_coef, Hill_par, gridDensity)
        #if nEq1 != 1:
        #print('at Hill coef', hill_coef, 'n. eqs is ', nEq1)
    except Exception as error:
        print('failed due to ', str(type(error).__name__ + "–" + str(error)))

    try:
        hill_coef = 100
        # print('Testing at hill coef = ', hill_coef)
        nEq100, _ = count_equilibria(f, hill_coef, Hill_par, gridDensity)
        print('at Hill coef', hill_coef, 'n. eqs is ', nEq100)
        # if nEq100 >= 0:
            #print('n. eqs is ', nEq100, '\n')
            #print('Trying with more details')
            #nEq100, _ = count_equilibria(f, 100, Hill_par, gridDensity)
            #print(nEq100, "\n")
    except Exception as error:
        print('failed due to ', str(type(error).__name__ + "–" + str(error)))

    continue

    try:
        SNParameters, otherBif = saddle_node_search(f, [1, 10, 20, 35, 50, 75, 100], Hill_par, ds, dsMinimum,
                                                         maxIteration=100, gridDensity=3, bisectionBool=True)
        if SNParameters == 0:
            n_saddles_idx = 0
        else:
            n_saddles_idx = np.max([len(SNParameters)-1, 2])  # more than 0 = many
        correlation_matrix[num_stable_FP - 1, n_saddles_idx] += 1

        printing_statement = 'Completion: ' + str(n_regions) + ' out of ' + str(niter) + ', region number ' + str(
            par_index)
        sys.stdout.write('\r' + printing_statement)
        sys.stdout.flush()
    except Exception as error:
        # turn an error into a warning and print the associated tag
        warnings.warn(str("An exception occurred:" + type(error).__name__ + "–" + str(error)))


