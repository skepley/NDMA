"""
Test and debug functions which map back and forth between NDMA and DSGRN parameters.

    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 6/28/22; Last revision: 6/28/22
"""
from ndma.basic_models.EMT_model import EMT
from ndma.parameter_generation.DSGRN_tools import *


def marcio_dict(par_idx, sampler):
    """Compare to Marcio parameter constructor using json tables"""

    dsgrn_par = parameter_graph_EMT.parameter(par_idx)
    D = EMT_network.size()
    L = np.zeros([D, D])
    U = np.zeros([D, D])
    T = np.zeros([D, D])

    # Get a dictionary from sample
    sample_dict = json.loads(sampler.sample(dsgrn_par))

    # Get values of L, U, and T from dictionary
    for key, value in sample_dict['Parameter'].items():
        # Get parameter (L, U, or T)
        par_type = key[0]
        # Extract variable names
        node_names = [name.strip() for name in key[2:-1].split('->')]
        node_indices = [EMT_network.index(node) for node in node_names]
        if par_type == 'L':
            L[tuple(node_indices)] = value
        elif par_type == 'U':
            U[tuple(node_indices)] = value
        else:  # T
            T[tuple(node_indices)] = value

    return L, U, T


# set up EMT network to test with
gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
# production parameters as variable
f = EMT(gammaVar, parameterVar)

# define the DSGRN network and pick out a multistable parameter
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
sampler = DSGRN.ParameterSampler(EMT_network)

nTest = 50
for jTest in range(nTest):
    par_idx = np.random.randint(parameter_graph_EMT.size())
    parameter = DSGRN.ParameterSampler(EMT_network).sample(
        DSGRN.ParameterGraph(EMT_network).parameter(par_idx))
    p = DSGRN_parameter_to_NDMA(EMT_network, par_idx, edgeCounts)
    # end DSGRN to NDMA parameter test

    region_test = NDMA_parameter_to_DSGRN(EMT_network, f, edgeCounts, 3,
                                          p)  # pass a dummy hill index and use NDMA functions to map into DSGRN region
    L, U, T = marcio_dict(par_idx, sampler)
    ground_truth = DSGRN.par_index_from_sample(parameter_graph_EMT, L, U,
                                               T)  # use Marcios builtin method to reconstruct parameter and map to DSGRN region
    assert region_test == ground_truth

print('parameter mappings work!')