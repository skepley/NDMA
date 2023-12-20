import DSGRN
import numpy as np
import json


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
    return all_pars, indices_domain, indices_input


def DSGRNpar_to_HillCont(L, T, U):
    """
    takes the standard DSGRN parameters (3 vectors) and returns NDMA parameter vector, together with some info on the network
    INPUT
    L, T, U : np.array with the ell, theta and u data
    OUTPUT
    all_pars : vector with all the parameter in NDMA format
    indices_domain : indices defining the network structure, in particular the source of each edge in the DSGRN order
    indices_input : indices defining the network structure, in particular the target of each edge in the DSGRN order
    """
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
    """
    given a parameter array and some network data, returns DSGRN parameters ell, theta and u
    INPUT
    par : NDMA vector of parameters
    indices_domain : vecotr with all the edges sources in DSGRN order
    indices_input : vector with all the edges targets in DSGRN order
    domain_size : dimension of the network
    OUTPUT
    L, T, U : arrays of ell, theta and u values
    """
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


def global_par_to_region(par, parameter_graph, indices_domain, indices_input, domain_size):
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
    L, U, T = HillContpar_to_DSGRN(par, indices_domain, indices_input, domain_size)
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
    if return_region_number<0:
        print('Problem because region number cannot be negative - probably problem higher up')
    return DSGRN_FP_per_index(return_region_number, parameter_graph)


