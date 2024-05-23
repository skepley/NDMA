import numpy as np
import scipy
import matplotlib.pyplot as plt
import graphviz
from create_dataset import create_dataset, distribution_sampler, generate_data_from_coefs, \
    normal_distribution_around_points
import json
from DSGRN_functionalities import *
from models.EMT_model import EMT, def_emt_hill_model
# from EMT_toolbox import *
from DSGRNcrawler import *

# size and name of dataset created
size_dataset = 10 ** 4
file_name = 'dataset_EMT_april24.npz'
graph_span = 10

print('This code creates a datset of size ', size_dataset, ' in file ', file_name, ' such that ')
print('the file has two information: the data (parameters of EMT) and the DSGRN region they belong to')
print('classified as 0 for monostable, 1 for bistable, 2 for other')
print('The regions are chosen in the first ', graph_span, 'DSGRN regions such that they are adjacent')
print('and each region is maximally enclosed with regions of its same stability')
print('i.e. the monostable region has many monostable regions around,',
      'and the bistable region has many bistable regions around')
""" 
The algorithm then creates the default points, and uses them as base to create a gaussian clou around them.
The Gaussian cloud is then 'optimised' by randomly tweaking its coefficients to better distribute the points it 
generates
"""

# create network from file
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)

possible_regions = np.array(range(graph_span))
monostable_regions = possible_regions[crawler.vec_is_monostable(possible_regions)]
mono_bistable_pairs = []

for par_index_i in monostable_regions:  # parameter_graph_EMT.size()
    bistable_list_i = crawler.bistable_neighbours(par_index_i)
    if bistable_list_i:
        mono_bistable_pairs.append([[par_index_i, bistable_index] for bistable_index in bistable_list_i])

num_parameters = parameter_graph_EMT.size()
num_candidates = len(mono_bistable_pairs)

print('Number of parameters in the parameter graph: ' + str(num_parameters))
print('Monostable parameters found in our search of the parameter graph: ' + str(len(monostable_regions)))
print('Of which, parameters with ad adjacent bistable region: ' + str(num_candidates))
print('All the pairs:\n', mono_bistable_pairs)


def score_many_monostable_and_many_bistable(pair_mono_bi):
    par_index_mono, par_index_bi = pair_mono_bi[0], pair_mono_bi[1]
    adjacent_nodes_mono = parameter_graph_EMT.adjacencies(par_index_mono)
    num_loc_monostable = sum(crawler.vec_is_monostable(adjacent_nodes_mono))
    score_monostable = num_loc_monostable / len(adjacent_nodes_mono)

    adjacent_nodes_bi = parameter_graph_EMT.adjacencies(par_index_bi)
    num_loc_bistable = sum([crawler.is_bistable(adjacent) for adjacent in adjacent_nodes_bi])
    score_bistable = num_loc_bistable / len(adjacent_nodes_bi)

    final_score = score_monostable ** 2 + score_bistable ** 2
    return final_score


# refine the search: to each good candidate count the number of monostable adjacent nodes / number of adjacent nodes and
# the same for the bistable node: we want as many monostable nodes close to the monostable node and as many bistable
# nodes near the bistable node
score_candidate = np.array([score_many_monostable_and_many_bistable(pair[0]) for pair in mono_bistable_pairs])
ranking = score_candidate.argsort()
best_pair = np.array(mono_bistable_pairs[ranking[-1]][0]) # highest score
monostable_region, bistable_region = best_pair[0], best_pair[1]
print('Chosen regions: ' + str(best_pair))

# sampling from each region
sampler = DSGRN.ParameterSampler(EMT_network)

monostable_parameternode = parameter_graph_EMT.parameter(monostable_region)
monostable_parameter = sampler.sample(monostable_parameternode)
print('monostable parameters from DSGRN as reference: \n', monostable_parameter)
bistable_parameternode = parameter_graph_EMT.parameter(bistable_region)
bistable_parameter = sampler.sample(bistable_parameternode)

f = def_emt_hill_model()
n_parameters_EMT = 42

# extract sheer data
bistable_pars, _, _ = from_string_to_Hill_data(bistable_parameter, EMT_network)
monostable_pars, indices_sources_EMT, indices_targets_EMT = from_string_to_Hill_data(monostable_parameter, EMT_network)

ND_sampler = distribution_sampler()
assign_region = par_to_region_wrapper(f, best_pair, parameter_graph_EMT, indices_sources_EMT, indices_targets_EMT)
print('Test: bistable and monostable regions', assign_region(np.array([bistable_pars, monostable_pars]).T))

# trying to get more points in region 1 (only points in region 0 otherwise)
# looking for "middle point" between region 0 and 1
# finding too many monostable, so moving the monostable point towards the bistable one
old_bistable_pars = bistable_pars
old_monostable_pars = monostable_pars
for i in range(10):
    middle_point = (bistable_pars+monostable_pars)/2
    if par_to_region(f, middle_point, best_pair, parameter_graph_EMT, indices_sources_EMT, indices_targets_EMT)==0:
        monostable_pars = middle_point
    else:
        print(i, 'iterations of bisection done to move the monostable pars closer to the bistable one')
        break
# Create initial distribution
Sigma, mu = normal_distribution_around_points(np.array([bistable_pars]), np.array([monostable_pars]))

# Create dataset
initial_coef = np.append(mu, Sigma.flatten())

bin_size = lambda vec: np.array([np.sum(vec == j) for j in range(2)])
score = lambda vec:  min(bin_size(vec))*len(bin_size(vec)) / np.size(vec)

best_coef = initial_coef
best_data = ND_sampler(initial_coef[:n_parameters_EMT], initial_coef[n_parameters_EMT:], 500)
best_parameter_region = assign_region(best_data)
best_score = score(best_parameter_region)

for i in range(100):
    random_coef = best_coef * (1 + np.random.rand(np.size(initial_coef)) * 0.05)
    data = ND_sampler(random_coef[:n_parameters_EMT], random_coef[n_parameters_EMT:], 500)
    parameter_region = assign_region(data)
    if score(parameter_region) > best_score:
        best_score = score(parameter_region)
        best_data = data
        best_parameter_region = parameter_region
        best_coef = random_coef

print('Final randomly generated score: ', best_score)
print('Number of data in regions: ', bin_size(best_parameter_region))

generate_data_from_coefs(file_name, best_coef, ND_sampler, assign_region, size_dataset, n_parameters_EMT)

# print('launching optimization')
# file_name = create_dataset(n_parameters, assign_region, n_parameter_region, size_dataset, file_name=file_name, initial_coef=initial_coef)
