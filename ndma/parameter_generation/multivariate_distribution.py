import matplotlib.pyplot as plt
from ndma.parameter_generation.DSGRN_tools import *


# let a and be be two vectors in high dimensions, we want to create a distribution that approximately give points along
# the segment [a,b]
from models.EMT_model import EMT


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (np.abs(w) > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
        else:
            print('The vectors are not linearly independent')
    return np.array(basis)


def normal_distribution_around_point(a):
    Sigma = 0.01*np.identity(np.size(a, 1))

    mu = a[0, :]
    return Sigma, mu


def normal_distribution_around_points(a, b):
    v1 = a - b
    lambda_1 = np.linalg.norm(a - b) / 2

    V = np.identity(np.size(a))
    index_info = np.argmax(v1)
    V[:, index_info] = v1
    V[:, [0, index_info]] = V[:, [index_info, 0]]

    Lambda = np.identity(np.size(a))
    Lambda = 10**-4 * lambda_1 * Lambda
    Lambda[0, 0] = lambda_1

    V = gram_schmidt(V.T)
    V = V.T

    Sigma = np.dot(np.dot(V, Lambda), V.T)

    mu = (a + b) / 2
    return Sigma, mu


def normal_distribution_around_many_points(points):
    size_subspace = np.size(points)[1]
    mean_point = np.mean(points, axis=1)
    average_distance = np.mean(points - mean_point, axis=1)

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


def test_multivar():
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
        ax.scatter([a[:, indeces_plot[0]], b[:, indeces_plot[0]], c[:, indeces_plot[0]]],
                   [a[:, indeces_plot[1]], b[:, indeces_plot[1]], c[:, indeces_plot[1]]], marker='*', s=100)
        ax.scatter(sample[:, indeces_plot[0]], sample[:, indeces_plot[1]], marker='o', s=4)

        plt.figure()
        fig = plt.gcf()
        ax = fig.gca()
        ax.scatter([a[:, indeces_plot[2]], b[:, indeces_plot[2]], c[:, indeces_plot[2]]],
                   [a[:, indeces_plot[3]], b[:, indeces_plot[3]], c[:, indeces_plot[3]]], marker='*', s=100)
        ax.scatter(sample[:, indeces_plot[2]], sample[:, indeces_plot[3]], marker='o', s=4)


def multivariate_normal_distributions(c1_vec, c2_vec, size):
    # par = np.zeros([len(c1_vec), size])
    mean = c1_vec
    dim = len(mean)
    cov = np.reshape(c2_vec, (dim, dim))
    x = np.random.multivariate_normal(mean, cov, size)
    par = np.abs(x).T
    # abs ensures it's positive
    return par


def generate_data_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters):
    """
    Takes the optimal coefficients and create a dataset out of them

    INPUT
    file_name       name of output file
    optimal_coef    optimal coefficients for the Fisher distribution
    sampler_global  way to sample from the correct distribution given the optimal parameters
    size_dataset    integer, size of the wanted dataset
    """

    data = sampler_global(optimal_coef[:n_parameters], optimal_coef[n_parameters:], size_dataset)
    parameter_region = assign_region(data)
    np.savez(file_name, optimal_coef=optimal_coef, data=data, parameter_region=parameter_region)
    return file_name


if __name__ == "__main__":

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

    for par_index in range(150):  # parameter_graph_EMT.size()
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
    grade_candidate = np.zeros((0, 2))
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
                num_loc_monostable = num_loc_monostable + 1
        ratio_monostable = num_loc_monostable / len(adjacent_nodes)

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
                num_loc_bistable = num_loc_bistable + 1
        ratio_bistable = num_loc_bistable / len(adjacent_nodes)
        grade_candidate = np.append(grade_candidate, [[ratio_monostable, ratio_bistable]], axis=0)

    best_candidate = np.argmax(grade_candidate[:, 0] ** 2 + grade_candidate[:, 1] ** 2)
    monostable_region = good_candidate[best_candidate][0]
    bistable_region = good_candidate[best_candidate][1]
    both_regions = np.array(good_candidate[best_candidate])
    print('Chosen regions: ' + str(both_regions))

    # edgeCount specific for the EMT network
    edgeCounts = [2, 2, 2, 1, 3, 2]
    p0 = DSGRN_parameter_to_NDMA(EMT_network, both_regions[0], edgeCounts)
    p1 = DSGRN_parameter_to_NDMA(EMT_network, both_regions[1], edgeCounts)

    # Create initial distribution
    Sigma, mu = normal_distribution_around_points(p0.flatten(), p1.flatten())

    # Create dataset
    n_parameter_region = 2
    size_dataset = 10**4
    file_name = 'dataset_multistable_EMT.npz'
    initial_coef = np.append(mu, Sigma.flatten())

    gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
    edgeCounts = [2, 2, 2, 1, 3, 2]
    parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
    # production parameters as variable
    f = EMT(gammaVar, parameterVar)

    assign_region = par_to_region_wrapper(EMT_network, f, edgeCounts, both_regions)

    NDMA_parameter_to_DSGRN(EMT_network, f, edgeCounts, np.nan, p0)

    data_sample = np.zeros((42, 1))
    data_sample = multivariate_normal_distributions(mu, Sigma.flatten(), 10**4)
    data_region = assign_region(data_sample)

    generate_data_from_coefs(file_name, initial_coef, multivariate_normal_distributions, assign_region, size_dataset, len(mu)) # done with range 10
    # file_name = create_dataset(n_parameters, assign_region, n_parameter_region, size_dataset, file_name=file_name, initial_coef=initial_coef)