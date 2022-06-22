"""
A collection of functions for translating data, parameters, and models between DSGRN combinatorial models and NDMA Hill models.
    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 6/22/22; Last revision: 6/22/22
"""
from hill_model import *
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


def parameter_from_DSGRN(dsgrnNetwork, parameterNodeIndex, edgeCount):
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
