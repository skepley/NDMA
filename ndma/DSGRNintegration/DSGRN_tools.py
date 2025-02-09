"""
A collection of functions for translating data, parameters, and models between DSGRN combinatorial models and NDMA Hill models.
    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 6/22/22
"""
from ndma.hill_model import *
from DSGRN import *
import re


def edge_parameter_from_DSGRN(edgeCount, parameter_string):
    """Input is a DSGRN parameter string which defines parameter values for L, U, T with edge labels
    defined by strings of the form Xi -> Xj. The output is the ordered parameter vector corresponding to the NDMA parameter
    ordering.
    NOTE: This function ONLY works for NDMA models which have ALL edge and decay parameters as variable. If you fix some NDMA parameters
    (other than the Hill coefficient) then you must handle this separately."""

    num_node = len(edgeCount)

    def order_matches(tuple_list):
        idx, unordered_parm = np.array(
            list(zip(*[(int(tup[1]) * num_node + int(tup[0]), float(tup[2])) for tup in tuple_list])))
        return unordered_parm[idx.argsort()]

    edge_parameter_match = re.compile(r"([LTU])\[X(\d)->X(\d)].*?:\s*(\d*.\d*)")  # define regex pattern to match
    match = edge_parameter_match.findall(parameter_string)  # get matches

    # get NDMA ordered parameters
    ell = order_matches([tup[1:] for tup in match if tup[0] == 'L'])
    theta = order_matches([tup[1:] for tup in match if tup[0] == 'T'])
    delta = order_matches([tup[1:] for tup in match if tup[0] == 'U']) - ell
    return np.array(list(zip(ell, delta, theta))).flatten()


def insert_trivial_decay(edgeCount, edge_parameter):
    insert_idx = 3 * np.cumsum(ezcat(0, edgeCount[:-1]))
    full_parameter = edge_parameter
    for edge_idx in range(len(insert_idx)):
        full_parameter = np.insert(full_parameter, edge_idx + insert_idx[edge_idx], 1)
    return full_parameter


def DSGRN_parameter_to_NDMA(dsgrnNetwork, parameterNodeIndex, edgeCount):
    """Return a parameter (without Hill coefficients) for an NDMA model associated with the DSGRN_network specified.
    This parameter will lie in the DSGRN parameter region specified by the given parameterNode which is the linear index
    for that region by DSGRN. edge_count is a vector of length (number of nodes) specifying the number of incoming edges
    to each node as ordered by the NDMA model.
    NOTE: This assumes that the DSGRN network model has nodes labelled as X_0, X_1, ..., X_(N-1). The regex search must
    be modified if the labels are different."""

    parameterString = DSGRN.ParameterSampler(dsgrnNetwork).sample(
        DSGRN.ParameterGraph(dsgrnNetwork).parameter(parameterNodeIndex))
    edgeParameter = edge_parameter_from_DSGRN(edgeCount, parameterString)
    fullParameter = insert_trivial_decay(edgeCount, edgeParameter)
    return fullParameter


def par_to_region_wrapper(dsgrnNetwork, hillModel, edgeCount, tracked_regions):
    def par_2_region(par_array):
        region_number = []
        for par in par_array.T:
            DSGRN_region = NDMA_parameter_to_DSGRN(dsgrnNetwork, hillModel, edgeCount, np.nan, par)
            if DSGRN_region in tracked_regions:
                region_number.append(np.where(tracked_regions == DSGRN_region)[0][0])
            else:
                region_number.append(len(tracked_regions))
        return np.array(region_number)
    return par_2_region


def network2matrix(dsgrnNetwork):
    """Return the adjacency matrix for a DSGRN network where A[i][j] is nonzero iff there is
      an edge from node j to node i. Note this is the transpose of the adjacency matrix for NDMA."""

    labelMatch = re.compile(r"(.*) :")  # regex for matching DSGRN network labels
    labels = labelMatch.findall(dsgrnNetwork.specification())  # get matches to identify node labels
    edgeStrings = dsgrnNetwork.specification().split('\n')  # read interactions for each node as strings
    adjDim = dsgrnNetwork.size()  # dimension of adjacency matrix
    adjMatrix = np.zeros([adjDim, adjDim])
    for (j, label) in enumerate(labels):
        adjMatrix[:, j] = [s in edgeStrings[j] and s != label for s in labels]

    return adjMatrix


def NDMA_parameter_to_DSGRN(dsgrnNetwork, hillModel, edgeCount, *parameter):
    """Convert a given NDMA parameter into a DSGRN parameter region index. Input should be a full parameter which can be
    parsed by the hillModel.parse_parameter method."""

    dim = hillModel.dimension
    parameterByCoordinate = hillModel.unpack_parameter(
        hillModel.parse_parameter(*parameter))  # concatenate all parameters into
    # a vector and unpack by coordinate

    # split up parameter vector into gamma, edge, and hill parameters
    gamma_pars = np.array([])
    edge_pars = []
    for j in range(dim):
        gamma, edge = np.split(parameterByCoordinate[j], [1])  # strip gamma, edge, and hill parameters for this coordinate
        gamma_pars = ezcat(gamma_pars, gamma)  # append gamma_j
        edge_pars.append(
            np.reshape(edge, (edgeCount[j], 4))[:, :-1])  # reshape as column matrix: [L, D, T] and discard hill

    adjMatrix = network2matrix(dsgrnNetwork).transpose()  # transpose to index in NDMA format (rows identify incoming edges)
    edge_pars = np.row_stack([edge for edge in edge_pars])  # stack and unpack L, D, T arrays
    ell, delta, theta = edge_pars.transpose()

    # initialize DSGRN adjacency matrices and write the parameter data correctly
    L = np.zeros([dim, dim])
    U = np.zeros([dim, dim])
    T = np.zeros([dim, dim])

    L[adjMatrix.nonzero()] = ell
    U[adjMatrix.nonzero()] = ell + delta
    T[adjMatrix.nonzero()] = theta

    # transpose back to match DSGRN adjacency format (columns identify incoming edges) and call DSGRN parameter lookup
    L = L.transpose()
    U = U.transpose()
    T = gamma_pars * T.transpose()  # by definition, T = gamma * theta

    return DSGRN.par_index_from_sample(DSGRN.ParameterGraph(dsgrnNetwork), L, U, T)



