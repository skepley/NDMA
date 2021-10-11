import numpy as np
import scipy
import matplotlib.pyplot as plt
from DSGRN import *
import graphviz


# let a and be be two vectors in high dimensions, we want to create a distribution that approximately give points along
# the segment [a,b]


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def normal_distribution_around_points(a, b):
    v1 = a - b
    lambda_1 = np.linalg.norm(a - b) / 2

    Lambda = np.identity(np.size(a, 1))
    Lambda = 0.001 * lambda_1 * Lambda
    Lambda[0, 0] = lambda_1

    V = np.identity(np.size(a, 1))
    V[:, 0] = v1
    V = gram_schmidt(V.T)
    V = V.T

    Sigma = np.dot(np.dot(V, Lambda), V.T)

    mu = (a[0, :] + b[0, :]) / 2
    return Sigma, mu

def normal_distribution_around_many_points(a, *args):
    size_subspace = len(args)
    central_point = a
    for vec in args:
        central_point = central_point + vec
    mean_point = central_point / (size_subspace+1)
    average_distance = a - mean_point
    for vec in args:
        average_distance = average_distance + vec - mean_point
    average_distance = average_distance / (size_subspace+1)

    V = np.identity(np.size(a, 1))
    V[:,0] = a - mean_point

    Lambda = np.identity(np.size(a, 1))
    lambda_1 = np.linalg.norm(a - mean_point) / 2
    Lambda = 0.000001 * average_distance * Lambda
    Lambda[0, 0] = lambda_1
    i = 1

    for vec in args:
        V[:, i] = vec
        Lambda[i, i] = np.linalg.norm(b - mean_point) / 2
        i += 1

    V = gram_schmidt(V.T)
    V = V.T

    Sigma = np.dot(np.dot(V, Lambda), V.T)
    mu = mean_point[0,:]
    return Sigma, mu



a = np.random.rand(1, 4)
b = np.random.rand(1, 4)
c = np.random.rand(1, 4)
make_figure = False

# Sigma, mu = normal_distribution_around_points(a, b)
Sigma, mu = normal_distribution_around_many_points(a, b, c)
sample = np.random.multivariate_normal(mu, Sigma, 300)

if make_figure:
    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter([a[:, 0], b[:, 0], c[:, 0]], [a[:, 1], b[:, 1], c[:, 1]], marker='*', s=100)
    ax.scatter(sample[:, 0], sample[:, 1], marker='o', s=4)

    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter([a[:, 2], b[:, 2], c[:, 2]], [a[:, 3], b[:, 3], c[:, 3]], marker='*', s=100)
    ax.scatter(sample[:, 2], sample[:, 3], marker='o', s=4)

# creat network from file
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

monostable_FP_parameters = []
bistable_FP_parameters = []
multistable_FP_parameters = []
good_candidate = []

for par_index in range(300): # parameter_graph_EMT.size()
    parameter = parameter_graph_EMT.parameter(par_index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    morse_nodes = range(morse_graph.poset().size())
    num_stable_FP = sum(1 for node in morse_nodes if isFP(node))
    if num_stable_FP == 1:
        monostable_FP_parameters.append(par_index)
        adjacent_nodes = parameter_graph_EMT.adjacencies(par_index)
        for adjacent in adjacent_nodes:
            parameter_adjacent = parameter_graph_EMT.parameter(adjacent)
            domain_graph_adjacent = DSGRN.DomainGraph(parameter_adjacent)
            morse_graph = DSGRN.MorseGraph(domain_graph_adjacent)
            morse_nodes_adjacent = range(morse_graph.poset().size())
            num_stable_FP_adjacent = sum(1 for node in morse_nodes_adjacent if isFP(node))
            if num_stable_FP_adjacent == 2:
                good_candidate.append((par_index, adjacent))
    elif num_stable_FP == 2:
        bistable_FP_parameters.append(par_index)
    elif num_stable_FP > 2:
        multistable_FP_parameters.append(par_index)

num_parameters = parameter_graph_EMT.size()
num_monostable_params = len(monostable_FP_parameters)
num_bistable_params = len(bistable_FP_parameters)
num_multistable_params = len(multistable_FP_parameters)
num_candidates = len(good_candidate)

print('Number of parameters in the parameter graph: ' + str(num_parameters))
print('Monostable parameters in the parameter graph: ' + str(num_monostable_params))
print('Bistable parameters in the parameter graph: ' + str(num_bistable_params))
print('Multistable parameters in the parameter graph: ' + str(num_multistable_params))
print('Good parameters in the parameter graph: ' + str(num_candidates))
print(good_candidate)


# refine the search: to each good candidate count the number of monostable adjacent nodes / number of adjacent nodes and
# the same for the bistable node: we want as many monostable nodes close to the monostable node and as many bistable
# nodes near the bistable node
grade_candidate = np.zeros((0,2))
for index in range(num_candidates):
    par_index_monostable = good_candidate[index][0]
    monostable_node = parameter_graph_EMT.parameter(par_index_monostable)
    adjacent_nodes = parameter_graph_EMT.adjacencies(par_index_monostable)
    num_loc_monostable = 0
    for adjacent in adjacent_nodes:
        parameter_adjacent = parameter_graph_EMT.parameter(adjacent)
        domain_graph_adjacent = DSGRN.DomainGraph(parameter_adjacent)
        morse_graph = DSGRN.MorseGraph(domain_graph_adjacent)
        morse_nodes_adjacent = range(morse_graph.poset().size())
        num_stable_FP_adjacent = sum(1 for node in morse_nodes_adjacent if isFP(node))
        if num_stable_FP_adjacent == 1:
            num_loc_monostable = num_loc_monostable+1
    ratio_monostable = num_loc_monostable/len(adjacent_nodes)

    par_index_bistable = good_candidate[index][1]
    bistable_node = parameter_graph_EMT.parameter(par_index_bistable)
    adjacent_nodes = parameter_graph_EMT.adjacencies(par_index_bistable)
    num_loc_bistable = 0
    for adjacent in adjacent_nodes:
        parameter_adjacent = parameter_graph_EMT.parameter(adjacent)
        domain_graph_adjacent = DSGRN.DomainGraph(parameter_adjacent)
        morse_graph = DSGRN.MorseGraph(domain_graph_adjacent)
        morse_nodes_adjacent = range(morse_graph.poset().size())
        num_stable_FP_adjacent = sum(1 for node in morse_nodes_adjacent if isFP(node))
        if num_stable_FP_adjacent == 2:
            num_loc_bistable = num_loc_bistable+1
    ratio_bistable = num_loc_bistable/len(adjacent_nodes)
    grade_candidate = np.append(grade_candidate,[[ratio_monostable, ratio_bistable]], axis = 0)

best_candidate = np.argmax(grade_candidate[:,0]**2 + grade_candidate[:,1]**2)
monostable_region = good_candidate[best_candidate][0]
bistable_region = good_candidate[best_candidate][1]

# # reality check  - are the regions correct?
# parameter_adjacent = parameter_graph_EMT.parameter(monostable_region)
# domain_graph_adjacent = DSGRN.DomainGraph(parameter_adjacent)
# morse_graph = DSGRN.MorseGraph(domain_graph_adjacent)
# morse_nodes_adjacent = range(morse_graph.poset().size())
# num_stable_FP_adjacent = sum(1 for node in morse_nodes_adjacent if isFP(node))
#
# parameter_adjacent = parameter_graph_EMT.parameter(bistable_region)
# domain_graph_adjacent = DSGRN.DomainGraph(parameter_adjacent)
# morse_graph = DSGRN.MorseGraph(domain_graph_adjacent)
# morse_nodes_adjacent = range(morse_graph.poset().size())
# num_stable_FP_adjacent = sum(1 for node in morse_nodes_adjacent if isFP(node))


# sampling from each region
sampler = DSGRN.ParameterSampler(EMT_network)

monostable_parameternode = parameter_graph_EMT.parameter(monostable_region)
monostable_parameter = sampler.sample(monostable_parameternode)

bistable_parameternode = parameter_graph_EMT.parameter(bistable_region)
bistable_parameter = sampler.sample(bistable_parameternode)

# extract sheer data??


# test of a given parameter is in either of our regions
def region_number(par):
    # if par is in monostable_parameternode:
    return 1
    # if par is in bistable_parameternode::
    # return 2
    # else:
    # return 0

# Create initial distribution
Sigma, mu = normal_distribution_around_points(a, b) # as soon as we have two parameters

# Create dataset
n_parameters = 54 # I think
n_parameter_region = 3
size_dataset = 10**4
file_name = 'dataset_EMT.npz'
initial_coef = np.array([mu, Sigma[:]])
create_dataset(n_parameters, region_number, n_parameter_region, size_dataset, file_name, initial_coef)


stophere
