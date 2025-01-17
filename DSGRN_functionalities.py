import warnings

import DSGRN
import numpy as np
import json
from ndma.hill_model import ezcat, is_vector


# notation: L, U, T
def from_region_to_deterministic_point(network, region_number: int):
    sampler = DSGRN.ParameterSampler(network)
    parameter_graph = DSGRN.ParameterGraph(network)

    parameternode = parameter_graph.parameter(region_number)
    parameter = sampler.sample(parameternode)

    # extract sheer data
    pars, indices_sources, indices_targets = from_string_to_Hill_data(parameter, network)
    return pars, indices_sources, indices_targets


def from_string_to_Hill_data(DSGRN_par_string, network):
    """
    extract Hill parameters from a DSGRN parameter string and the network info, returns additional network info
    INPUT
    DSGRN_par_string : string of DSGRN parameters
    network : DSGRN network
    OUTPUT
    all_pars : vector with all the parameter in NDMA format
    indices_domain : indices defining the network structure, in particular the source of each edge in the DSGRN order
    indices_input : indices defining the network structure, in particular the target of each edge in the DSGRN order
    """
    # OLD:
    # def from_string_to_Hill_data(DSGRN_par_string, domain_size, network, parameter_graph, region):
    L, U, T = from_string_to_DSGRN_data(DSGRN_par_string, network)

    all_pars, indices_domain, indices_input = DSGRNpar_to_HillCont(L, U, T)

    return all_pars, indices_domain, indices_input


def from_string_to_DSGRN_data(DSGRN_par_string, network):
    """
    extract Hill parameters from a DSGRN parameter string and the network info, returns additional network info
    INPUT
    DSGRN_par_string : string of DSGRN parameters
    network : DSGRN network
    OUTPUT
    all_pars : vector with all the parameter in NDMA format
    indices_domain : indices defining the network structure, in particular the source of each edge in the DSGRN order
    indices_input : indices defining the network structure, in particular the target of each edge in the DSGRN order
    """
    # OLD:
    # def from_string_to_Hill_data(DSGRN_par_string, domain_size, network, parameter_graph, region):
    D = network.size()
    L = np.zeros([D, D])
    U = np.zeros([D, D])
    T = np.zeros([D, D])
    indices_domain = []
    indices_input = []
    sample_dict = json.loads(DSGRN_par_string)
    for key, value in sample_dict['Parameter'].items():
        # Get parameter (L, T, or U)
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

    return L, U, T


def DSGRNpar_to_HillCont(L, U, T):
    """
    takes the standard DSGRN parameters (3 vectors) and returns NDMA parameter vector, together with some info on the network
    INPUT
    L, U, T : np.array with the ell, theta and u data
    OUTPUT
    all_pars : vector with all the parameter in NDMA format
    indices_domain : indices defining the network structure, in particular the source of each edge in the DSGRN order
    indices_input : indices defining the network structure, in particular the target of each edge in the DSGRN order
    """
    indices = np.array(np.nonzero(L))
    indices_domain = indices[0, :]
    indices_input = indices[1, :]

    domain_size = np.shape(L)[0]
    gamma = np.ones(domain_size)
    delta = U - L

    all_pars = np.array([])
    for column in range(domain_size):
        column_pars = ezcat(*list(zip(L[:, column], delta[:, column], T[:, column])))
        column_pars = column_pars[np.nonzero(column_pars)]

        all_pars = np.append(all_pars, gamma[column])
        all_pars = np.append(all_pars, column_pars)

    return all_pars, indices_domain, indices_input


def HillContpar_to_DSGRN(hillmodel, par, indices_sources, indices_target):
    """
    given a parameter array and some network data, returns DSGRN parameters ell, theta and u
    INPUT
    hillmodel: NDMA hill model
    par : NDMA vector of parameters
    indices_domain : vector with all the edges sources in DSGRN order
    indices_input : vector with all the edges targets in DSGRN order
    OUTPUT
    L, U, T : arrays of ell, theta and u values
    """
    domain_size = hillmodel.dimension
    number_of_edges = int((len(par) - domain_size) / 3)

    par = np.insert(par, hillmodel.hillInsertionIndex, np.nan)
    # add the hill coefficient in all the right spots for the hill model to work it out

    param_by_coords = hillmodel.unpack_parameter(par)
    Gamma = [param_by_coords[i][0] for i in range(len(param_by_coords))]

    L = np.zeros((domain_size, domain_size))  # equation, input
    T = np.zeros((domain_size, domain_size))
    Delta = np.zeros((domain_size, domain_size))

    # creating the correct indices for the DSGRN matrices
    index_reordering = np.argsort(indices_sources)
    indices_sources = np.array(indices_sources)
    indices_target = np.array(indices_target)
    indices_sources = indices_sources[index_reordering]
    indices_target = indices_target[index_reordering]

    all_ell, all_delta, all_theta = np.array([]), np.array([]), np.array([])
    for coord_index in range(len(param_by_coords)):
        gamma, list_of_component_pars = hillmodel.coordinates[coord_index].parse_parameters(
            param_by_coords[coord_index])
        for j in range(len(list_of_component_pars)):
            ell, delta, theta, hillCoefficient = hillmodel.coordinates[coord_index].productionComponents[
                j].curry_parameters(
                list_of_component_pars[j])
            all_ell = np.append(all_ell, ell)
            all_delta = np.append(all_delta, delta)
            all_theta = np.append(all_theta, theta)
    indices = list(zip(indices_sources, indices_target))
    indices.sort(key=lambda x: x[1])
    indices = np.array(indices)
    L[indices[:, 0], indices[:, 1]] = all_ell
    T[indices[:, 0], indices[:, 1]] = all_theta
    Delta[indices[:, 0], indices[:, 1]] = all_delta

    for i in range(domain_size):
        T[:, i] = T[:, i] * Gamma[i]
    U = L + Delta
    return L, U, T


def par_to_region(hillmodel, par, regions_array, parameter_graph, indices_sources, indices_target):
    """
    associate to a NDMA parameter the corresponding DSGRN region number
    INPUT
    par : vector of NDMA parameters
    regions_array : list of possible regions to chose from
    paramter_graph : DSGRN parameter graph
    indices_domain : vecotr with all the edges sources in DSGRN order
    indices_input : vector with all the edges targets in DSGRN order
    domain_size : dimension of the network
    OUTPUT

    """
    extended_region_number = global_par_to_region(hillmodel, par, parameter_graph, indices_sources, indices_target)
    if np.shape(regions_array) == (): # special case for only one region
        if extended_region_number == regions_array:
            return 0
        else:
            return 1
    restricted_region_number = np.where(extended_region_number == regions_array)
    if np.shape(restricted_region_number)[1] == 0: # if no match, return length of region array
        return len(regions_array)
    region_number = restricted_region_number[0][0]
    return region_number


def par_to_region_wrapper(hillmodel, regions_array, parameter_graph, indices_domain, indices_input):
    regions_array = np.array(regions_array)

    def par_2_region(par_array):
        region_number = []
        for par in par_array.T:
            region_number.append(
                par_to_region(hillmodel, par, regions_array, parameter_graph, indices_domain, indices_input))
        return np.array(region_number)

    return par_2_region


def global_par_to_region(hillmodel, par, parameter_graph, indices_sources, indices_target):
    """
    takes a NDMA parameter and return a DSGNR region number
    INPUT
    par : vector of NDMA parameters
    parameter_graph : DSGRN parameter graph
    indices_domain : vecotr with all the edges sources in DSGRN order
    indices_input : vector with all the edges targets in DSGRN order
    domain_size : dimension of the network
    OUTPUT
    return_region_number : DSGRN region number
    """
    L, U, T = HillContpar_to_DSGRN(hillmodel, par, indices_sources, indices_target)
    return_region_number = DSGRN.par_index_from_sample(parameter_graph, L, U, T)
    return return_region_number


def DSGRN_FP_per_index(par_index, parameter_graph):
    """
    returns the number of Fixed Points in a given parameter region in DSGRN
    INPUT
    par_index : the parameter region index
    parameter_graph : the DSGRN parameter graph
    OUTPUT
    num_stable_FP : int number of stable fixed points
    """
    parameter = parameter_graph.parameter(par_index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    isFP = lambda morse_node: morse_graph.annotation(morse_node)[0].startswith('FP')
    morse_nodes = range(morse_graph.poset().size())
    num_stable_FP = sum(1 for node in morse_nodes if isFP(node))
    return num_stable_FP


def par_to_n_eqs(par, parameter_graph, indices_domain, indices_input, domain_size):
    """
    given a NDMA parameter, returns the expected number of fixed points according to DSGRN
    INPUT
    par : NDMA parameter
    parameter_graph : DSGRN parameter graph
    indices_domain : vecotr with all the edges sources in DSGRN order
    indices_input : vector with all the edges targets in DSGRN order
    domain_size : dimension of the network
    OUTPUT
    num_stable_FP : int number of stable fixed points
    """
    L, U, T = HillContpar_to_DSGRN(par, indices_domain, indices_input, domain_size)
    return_region_number = DSGRN.par_index_from_sample(parameter_graph, L, U, T)
    if return_region_number < 0:
        print('Problem because region number cannot be negative - probably problem higher up')
    return DSGRN_FP_per_index(return_region_number, parameter_graph)


def random_in_region(par_index, network, parameter_graph, size_out, variance=[]):
    """
    creates a random sample of parameters that all lie within the given DSGRN region index
    INPUT
    par_index : DSGRN region number
    network, parameter_graph : DSGRN networks and parameter graphs
    size_out : expected dimension of the output sample
    variance (optional) : wished for variance used to generate the point cloud
    OUTPUT
    par_region : array of NDMA parameter values in the given region
    """
    domain_size = network.size()
    parameternode = parameter_graph.parameter(par_index)
    sampler = DSGRN.ParameterSampler(network)
    p = sampler.sample(parameternode)
    par_NDMA, indices_domain, indices_input = from_string_to_Hill_data(p, network)

    def par_in_region(par):
        return global_par_to_region(par, parameter_graph, indices_domain, indices_input, domain_size) == par_index

    if not variance:
        variance = 0.02 * np.eye(len(par_NDMA))
    if is_vector(variance):
        variance = np.diag(variance)

    par_region = padded_filter_multivariate(par_NDMA, variance, size_out, lambda x: par_in_region(x))
    return par_region


def random_in_2regions(par_index1, par_index2, network, parameter_graph, size_out):
    """
    creates a random sample of parameters that all lie within the given DSGRN region index
    INPUT
    par_index1, par_index2 : DSGRN region number
    network, parameter_graph : DSGRN networks and parameter graphs
    size_out : expected dimension of the output sample
    OUTPUT
    par_region : array of NDMA parameter values in the two given regions
    """
    domain_size = network.size()
    parameternode1 = parameter_graph.parameter(par_index1)
    parameternode2 = parameter_graph.parameter(par_index2)
    sampler = DSGRN.ParameterSampler(network)

    p1 = sampler.sample(parameternode1)
    p2 = sampler.sample(parameternode2)

    par1, indices_domain, indices_input = from_string_to_Hill_data(p1, network)
    par2, _, _ = from_string_to_Hill_data(p2, network)

    mean = (par1 + par2) / 2
    main_axes = (par1 - par2) / 2
    distance_of_region_centers = np.linalg.norm(main_axes)
    variance = 0.01 * np.eye(len(mean))
    rescale_factor = distance_of_region_centers / mean[0]
    variance[0, :] = rescale_factor * main_axes

    def par_in_2regions(par):
        index = global_par_to_region(par, parameter_graph, indices_domain, indices_input, domain_size)
        return index == par_index1 or index == par_index2

    par_region = padded_filter_multivariate(mean, variance, size_out, par_in_2regions)

    return par_region


def padded_filter_multivariate(mean, variance, size_out, lambda_function):
    """
    this function returns an array of vectors sampled from the multivariate normal distribution with given mean and
    variance that satisfies the given boolean lambda function
    INPUT
    mean, variance : for the normal multivariate distribution
    size_out : size of the filtered point cloud - if achieved
    lambda_function : boolean function used for filtering
    OUTPUT
    point_cloud : point cloud satisfying the given boolean function
    """
    par_region = filtered_multivariate(mean, variance, size_out * 10, lambda_function)
    iter_rand = 0
    while np.size(par_region, 0) < size_out and iter_rand < 5:
        iter_rand += 1
        variance = variance * 0.95  # decrease variance to increase hit rate, hopefully
        par_region = np.append(par_region, filtered_multivariate(mean, variance, size_out * 10, lambda_function),
                               axis=0)
    par_region = par_region[:size_out, :]
    if np.size(par_region, 0) < size_out:
        warnings.warn("default", print('Number of points created within the region in lower than expected\n',
                                       'Number of expected points: ', size_out, '\nNumber of points: ',
                                       len(par_region)))
    return par_region


def filtered_multivariate(mean, variance, size, lambda_function):
    """
    this function returns an array of vectors sampled from the multivariate normal distribution with given mean and
    variance that satisfies the given boolean lambda function
    INPUT
    mean, variance : for the normal multivariate distribution
    size : size of the point cloud (NOT of the final result)
    lambda_function : boolean function used for filtering
    OUTPUT
    point_cloud : point cloud satisfying the given boolean function
    """
    random_cloud = np.abs(
        np.random.multivariate_normal(mean, variance, size))  # only the positive quadrant is considered
    point_cloud = random_cloud[[lambda_function(vec) for vec in random_cloud]]
    return point_cloud


def filter_region_wrt_morse_graph(iterative_regions, parameter_graph, filter_condition):
    """for all parameter regions given, return the ones satistying the filter condition
    """
    accepted_regions = []
    for par_index in iterative_regions:
        parameter = parameter_graph.parameter(par_index)
        domain_graph = DSGRN.DomainGraph(parameter)
        morse_graph = DSGRN.MorseGraph(domain_graph)
        if filter_condition(morse_graph):
            accepted_regions.append(par_index)
    return accepted_regions


def filter_region_wrt_index(iterative_regions, parameter_graph, filter_condition):
    """for all parameter regions given, return the ones satistying the filter condition on the region index
    """
    accepted_regions = []
    for par_index in iterative_regions:
        if filter_condition(par_index):
            accepted_regions.append(par_index)
    return accepted_regions


def compute_rank_region_wrt_morse_graph(iterative_regions, parameter_graph, ranking_function):
    """for all parameter regions given, return their rank (maintaining order)
    """
    rank_regions = []
    for par_index in iterative_regions:  # parameter_graph_EMT.size()
        parameter = parameter_graph.parameter(par_index)
        domain_graph = DSGRN.DomainGraph(parameter)
        morse_graph = DSGRN.MorseGraph(domain_graph)
        rank_regions.append(ranking_function(morse_graph))
    return rank_regions


def rank_region_wrt_morse_graph(iterative_regions, parameter_graph, ranking_function):
    """for all parameter regions given, return their rank (maintaining order)
    """
    rank_regions = compute_rank_region_wrt_morse_graph(iterative_regions, parameter_graph, ranking_function)
    order = rank_regions.argsort()
    sorted_regions = iterative_regions[order]
    return sorted_regions
