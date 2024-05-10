import numpy as np
import scipy
import matplotlib.pyplot as plt
import graphviz
from create_dataset import create_dataset, distribution_sampler, generate_data_from_coefs
import json
from DSGRN_functionalities import *
from models.EMT_model import EMT


# let a and b be two vectors in high dimensions, we want to create a distribution that approximately give points along
# the segment [a,b]

def normal_distribution_around_points(a, b):
    v1 = a - b
    lambda_1 = np.linalg.norm(a - b) / 2

    V = np.identity(np.size(a, 1))
    index_info = np.argmax(v1)
    V[:, index_info] = v1
    V[:, [0, index_info]] = V[:, [index_info, 0]]

    Lambda = np.identity(np.size(a, 1))
    Lambda = 10 ** -4 * lambda_1 * Lambda
    Lambda[0, 0] = lambda_1

    V, _ = np.linalg.qr(V.T).T

    Sigma = np.dot(np.dot(V, Lambda), V.T)

    mu = (a[0, :] + b[0, :]) / 2
    return Sigma, mu


def normal_distribution_around_many_points(a, *args):
    size_subspace = len(args)
    central_point = a
    for vec in args:
        central_point = central_point + vec
    mean_point = central_point / (size_subspace + 1)
    average_distance = a - mean_point
    for vec in args:
        average_distance = average_distance + vec - mean_point
    average_distance = average_distance / (size_subspace + 1)

    V = np.identity(np.size(a, 1))
    V[:, 0] = a - mean_point

    Lambda = np.identity(np.size(a, 1))
    lambda_1 = np.linalg.norm(a - mean_point) / 2
    Lambda = 0.0001 * average_distance * Lambda
    Lambda[0, 0] = 0.01 * lambda_1
    i = 1

    for vec in args:
        V[:, i] = vec
        Lambda[i, i] = 0.01 * lambda_1
        i += 1

    V, _ = np.linalg.qr(V.T).T

    Sigma = np.dot(np.dot(V, Lambda), V.T)
    mu = mean_point[0, :]
    return Sigma, mu


# size and name of dataset created
size_dataset = 10 ** 4
file_name = 'dataset_EMT_april24.npz'
graph_span = 100

print('This code creates a datset of size ', size_dataset, ' in file ', file_name, ' such that ')
print('the file has two information: the data (parameters of EMT) and the DSGRN region they belong to')
print('classified as 0 for monostable, 1 for bistable, 2 for other')
print('The regions are chosen in the first ', graph_span, 'DSGRN regions such that they are adjacent')
print('and each region is maximally enclosed with regions of its same stability')
print('i.e. the monostable region has many monostable regions around, and the bistable region has many bistable regions around')
""" 
The algorithm then creates the default points, and uses them as base to create a gaussian clou around them.
The Gaussian cloud is then 'optimised' by randomly tweaking its coefficients to better distribute the points it 
generates
"""


# create network from file
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)

num_monostable_parameters = 0
mono_bistable_pairs = []

isFP = lambda morse_node: morse_graph.annotation(morse_node)[0].startswith('FP')
is_monostable = lambda morse_graph: sum(1 for node in range(morse_graph.poset().size()) if isFP(node)) == 1
is_bistable = lambda morse_graph: sum(1 for node in range(morse_graph.poset().size()) if isFP(node)) == 2


def morse_graph_from_index(index):
    parameter = parameter_graph_EMT.parameter(index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    return morse_graph


for par_index in graph_span:  # parameter_graph_EMT.size()
    morse_graph = morse_graph_from_index(par_index)
    if is_monostable(morse_graph):
        num_monostable_parameters += 1
        adjacent_nodes = parameter_graph_EMT.adjacencies(par_index)
        for adjacent in adjacent_nodes:
            morse_graph = morse_graph_from_index(adjacent)
            if is_bistable(morse_graph):
                mono_bistable_pairs.append((par_index, adjacent))

num_parameters = parameter_graph_EMT.size()
num_candidates = len(mono_bistable_pairs)

print('Number of parameters in the parameter graph: ' + str(num_parameters))
print('Monostable parameters found in our search of the parameter graph: ' + str(num_monostable_parameters))
print('Of which, parameters with ad adjacent bistable region: ' + str(num_candidates))
print('All the pairs:\n', mono_bistable_pairs)

# refine the search: to each good candidate count the number of monostable adjacent nodes / number of adjacent nodes and
# the same for the bistable node: we want as many monostable nodes close to the monostable node and as many bistable
# nodes near the bistable node
grade_candidate = np.zeros((0, 2))
for index in range(num_candidates):
    # check monostable regions around monostable region M
    par_index_monostable = mono_bistable_pairs[index][0]
    monostable_node = parameter_graph_EMT.parameter(par_index_monostable)
    adjacent_nodes = parameter_graph_EMT.adjacencies(par_index_monostable)
    num_loc_monostable = sum(is_monostable(morse_graph_from_index(adjacent)) for adjacent in adjacent_nodes)
    ratio_monostable = num_loc_monostable / len(adjacent_nodes)

    # check bistable regions around bistable region B
    par_index_bistable = mono_bistable_pairs[index][1]
    bistable_node = parameter_graph_EMT.parameter(par_index_bistable)
    adjacent_nodes = parameter_graph_EMT.adjacencies(par_index_bistable)
    num_loc_bistable = sum(is_bistable(morse_graph_from_index(adjacent)) for adjacent in adjacent_nodes)
    ratio_bistable = num_loc_bistable / len(adjacent_nodes)

    grade_candidate = np.append(grade_candidate, [[ratio_monostable, ratio_bistable]], axis=0)

best_candidate = np.argmax(grade_candidate[:, 0] ** 2 + grade_candidate[:, 1] ** 2)
monostable_region = mono_bistable_pairs[best_candidate][0]
bistable_region = mono_bistable_pairs[best_candidate][1]
both_regions = np.array(mono_bistable_pairs[best_candidate])
print('Chosen regions: ' + str(both_regions))

# sampling from each region
sampler = DSGRN.ParameterSampler(EMT_network)

monostable_parameternode = parameter_graph_EMT.parameter(monostable_region)
monostable_parameter = sampler.sample(monostable_parameternode)
print('monostable parameters from DSGRN as reference: \n', monostable_parameter)
bistable_parameternode = parameter_graph_EMT.parameter(bistable_region)
bistable_parameter = sampler.sample(bistable_parameternode)

# define the EMT hill model
gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)
domain_size_EMT = 6
n_parameters = 42

# extract sheer data
bistable_pars, _, _ = from_string_to_Hill_data(bistable_parameter, EMT_network)
monostable_pars, indices_sources_EMT, indices_targets_EMT = from_string_to_Hill_data(monostable_parameter, EMT_network)

# Create initial distribution
Sigma, mu = normal_distribution_around_points(np.reshape(bistable_pars, (1, -1)), np.reshape(monostable_pars, (1, -1)))

# Create dataset
initial_coef = np.append(mu, Sigma.flatten())
assign_region = par_to_region_wrapper(f, both_regions, parameter_graph_EMT, indices_sources_EMT, indices_targets_EMT)

ND_sampler = distribution_sampler()

data_sample = ND_sampler(mu, Sigma.flatten(), 10 ** 4)
data_region = assign_region(data_sample)

bin_size = lambda vec: np.array([np.sum(vec == j) for j in range(np.max(vec))])
score = lambda vec: 1 - (np.max(bin_size(vec)) - min(bin_size(vec))) / np.size(vec)

best_coef = initial_coef
best_data = ND_sampler(initial_coef[:n_parameters], initial_coef[n_parameters:], 500)
best_parameter_region = assign_region(best_data)
best_score = score(best_parameter_region)

for i in range(100):
    random_coef = initial_coef + np.random.rand(np.size(initial_coef)) * 0.3
    data = ND_sampler(random_coef[:n_parameters], random_coef[n_parameters:], 500)
    parameter_region = assign_region(data)
    if score(parameter_region) > best_score:
        best_score = score(parameter_region)
        best_data = data
        best_parameter_region = parameter_region
        best_coef = random_coef

print('Final randomly generated score: ', best_score)
print('Number of data in regions: ', bin_size(best_parameter_region))

generate_data_from_coefs(file_name, best_coef, ND_sampler, assign_region, size_dataset, n_parameters)

# print('launching optimization')
# file_name = create_dataset(n_parameters, assign_region, n_parameter_region, size_dataset, file_name=file_name, initial_coef=initial_coef)

