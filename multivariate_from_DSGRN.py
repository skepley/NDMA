import numpy as np
import scipy
import matplotlib.pyplot as plt
import DSGRN
import graphviz
from create_dataset import create_dataset, region_sampler, generate_data_from_coefs
import json


# let a and be be two vectors in high dimensions, we want to create a distribution that approximately give points along
# the segment [a,b]


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (np.abs(w) > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
        else:
            disp('The vectors are not linearly independent')
    return np.array(basis)


def normal_distribution_around_points(a, b):
    v1 = a - b
    lambda_1 = np.linalg.norm(a - b) / 2

    V = np.identity(np.size(a, 1))
    index_info = np.argmax(v1)
    V[:, index_info] = v1
    V[:, [0, index_info]] = V[:, [index_info, 0]]

    Lambda = np.identity(np.size(a, 1))
    Lambda = 10**-4 * lambda_1 * Lambda
    Lambda[0, 0] = lambda_1

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
    Lambda = 0.0001 * average_distance * Lambda
    Lambda[0, 0] = 0.01*lambda_1
    i = 1

    for vec in args:
        V[:, i] = vec
        Lambda[i, i] = 0.01*lambda_1
        i += 1

    V = gram_schmidt(V.T)
    V = V.T

    Sigma = np.dot(np.dot(V, Lambda), V.T)
    mu = mean_point[0,:]
    return Sigma, mu


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


def DSGRNpar_to_HillCont(L, T, U):
    gamma = np.ones(shape=(1, np.shape(L)[0]))
    indices = np.array(np.nonzero(L))
    indices_domain = indices[0, :]
    indices_input = indices[1, :]
    delta = U - L
    ell_non_zero = L[np.nonzero(L)]
    theta_non_zero = T[np.nonzero(T)]
    delta_non_zero = delta[np.nonzero(delta)]
    all_pars = np.append(gamma, np.append(ell_non_zero, np.append(theta_non_zero, delta_non_zero)))
    return all_pars, indices_domain, indices_input


def HillContpar_to_DSGRN(par, indices_domain, indices_input, domain_size):
    data_size = int((len(par) - domain_size)/3)
    gamma = par[0:domain_size]
    L = np.zeros((domain_size, domain_size))  # equation, input
    T = np.zeros((domain_size, domain_size))
    delta = np.zeros((domain_size, domain_size))
    begin_L = domain_size
    end_L = begin_L + data_size
    begin_T = begin_L + data_size
    end_T = begin_T + data_size
    begin_U = begin_T + data_size
    end_U = begin_U + data_size
    index_reordering = np.argsort(indices_domain)
    indices_domain = np.array(indices_domain)
    indices_input = np.array(indices_input)
    indices_domain = indices_domain[index_reordering]
    indices_input = indices_input[index_reordering]
    L[indices_domain, indices_input] = par[begin_L:end_L]
    T[indices_domain, indices_input] = par[begin_T:end_T]
    for i in range(np.shape(T)[0]):
        T[i, :] = T[i, :]/gamma[i]
    delta[indices_domain, indices_input] = par[begin_U:end_U]
    U = L + delta
    return L, U, T


def par_to_region(par, regions_array, parameter_graph, indices_domain, indices_input, domain_size):
    L, U, T = HillContpar_to_DSGRN(par, indices_domain, indices_input, domain_size)
    extended_region_number = DSGRN.par_index_from_sample(parameter_graph, L, U, T)
    restricted_region_number = np.where(extended_region_number == regions_array)
    if np.shape(restricted_region_number)[1] == 0:
        return len(regions_array)
    region_number = restricted_region_number[0][0]
    return region_number


def par_to_region_wrapper(regions_array, parameter_graph, indices_domain, indices_input, domain_size):
    def par_2_region(par_array):
        region_number = []
        for par in par_array.T:
            region_number.append(par_to_region(par, regions_array, parameter_graph, indices_domain, indices_input, domain_size))
        return np.array(region_number)
    return par_2_region


a = np.random.rand(1, 42)
b = np.random.rand(1, 42)
c = np.random.rand(1, 42)
make_figure = True

# Sigma, mu = normal_distribution_around_points(a, b)
Sigma, mu = normal_distribution_around_many_points(a, b, c)
sample = np.random.multivariate_normal(mu, Sigma, 300)

if make_figure:
    indeces_plot = [11, 22, 33, 40]
    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter([a[:, indeces_plot[0]], b[:, indeces_plot[0]], c[:, indeces_plot[0]]], [a[:, indeces_plot[1]], b[:, indeces_plot[1]], c[:, indeces_plot[1]]], marker='*', s=100)
    ax.scatter(sample[:, indeces_plot[0]], sample[:, indeces_plot[1]], marker='o', s=4)

    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter([a[:, indeces_plot[2]], b[:, indeces_plot[2]], c[:, indeces_plot[2]]], [a[:, indeces_plot[3]], b[:, indeces_plot[3]], c[:, indeces_plot[3]]], marker='*', s=100)
    ax.scatter(sample[:, indeces_plot[2]], sample[:, indeces_plot[3]], marker='o', s=4)

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

monostable_FP_parameters = []
bistable_FP_parameters = []
multistable_FP_parameters = []
good_candidate = []

for par_index in range(500):  # parameter_graph_EMT.size()
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

best_candidate = np.argmax(grade_candidate[:, 0]**2 + grade_candidate[:, 1]**2)
monostable_region = good_candidate[best_candidate][0]
bistable_region = good_candidate[best_candidate][1]
both_regions = np.array(good_candidate[best_candidate])
print('Chosen regions: ' + str(both_regions))

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
print(monostable_parameter)
bistable_parameternode = parameter_graph_EMT.parameter(bistable_region)
bistable_parameter = sampler.sample(bistable_parameternode)

# extract sheer data??
domain_size_EMT = 6
bistable_pars, _, _ = from_string_to_Hill_data(bistable_parameter, domain_size_EMT, EMT_network, parameter_graph_EMT, bistable_region)
monostable_pars, indices_domain_EMT, indices_input_EMT = from_string_to_Hill_data(monostable_parameter, domain_size_EMT, EMT_network, parameter_graph_EMT, monostable_region)

L, U, T = HillContpar_to_DSGRN(monostable_pars, indices_domain_EMT, indices_input_EMT, domain_size_EMT)
success = (DSGRN.par_index_from_sample(parameter_graph_EMT, L, U, T) == monostable_region)

if not success:
    raise ValueError('Debugging error')

success = (par_to_region(bistable_pars, both_regions, parameter_graph_EMT, indices_domain_EMT, indices_input_EMT, domain_size_EMT) == 1)
if not success:
    raise ValueError('Debugging error')
success = (par_to_region(monostable_pars, both_regions, parameter_graph_EMT, indices_domain_EMT, indices_input_EMT, domain_size_EMT) == 0)
if not success:
    raise ValueError('Debugging error')

# Create initial distribution
Sigma, mu = normal_distribution_around_points(np.reshape(bistable_pars, (1, -1)), np.reshape(monostable_pars, (1, -1)))

# Create dataset
n_parameters = len(bistable_pars)
n_parameter_region = 3
size_dataset = 10**4
file_name = 'dataset_EMT.npz'
initial_coef = np.append(mu, Sigma.flatten())
assign_region = par_to_region_wrapper(both_regions, parameter_graph_EMT, indices_domain_EMT, indices_input_EMT, domain_size_EMT)

sampler_global = region_sampler()
data_sample = np.zeros((42,2))
data_sample[:, 0] = bistable_pars
data_sample[:, 1] = monostable_pars
ar = assign_region(data_sample)
if ar[0] != 1:
    raise ValueError('Debugging error')
if ar[1] != 0:
    raise ValueError('Debugging error')

data_sample = sampler_global(mu, Sigma.flatten(), 10**4)
data_region = assign_region(data_sample)

# vector data_region[1:size_sample], such that data_region[i] tells us which region par[:, i] belongs to
# data_region goes from 0 to n_parameter_region -1
counter = np.zeros(n_parameter_region)
for iter_loc in range(n_parameter_region):
    counter[iter_loc] = np.count_nonzero(data_region == iter_loc)
score = 1 - np.min(counter)/np.max(counter)

make_figure2 = True
if make_figure2:
    indeces_plot = [11, 22, 33, 40]
    a = bistable_pars
    b = monostable_pars
    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter(data_sample[indeces_plot[0], :], data_sample[indeces_plot[1], :], marker='o', s=4)
    ax.scatter([a[indeces_plot[0]], b[indeces_plot[0]]], [a[indeces_plot[1]], b[indeces_plot[1]]], marker='*', s=100)
    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter(data_sample[indeces_plot[2], :], data_sample[indeces_plot[3], :], marker='o', s=4)
    ax.scatter([a[indeces_plot[2]], b[indeces_plot[2]]], [a[indeces_plot[3]], b[indeces_plot[3]]], marker='*', s=100)

sampler_global = region_sampler()
generate_data_from_coefs(file_name, initial_coef, sampler_global, assign_region, size_dataset, n_parameters) # done with range 10

file_name = create_dataset(n_parameters, assign_region, n_parameter_region, size_dataset, file_name=file_name, initial_coef=initial_coef)

#stophere