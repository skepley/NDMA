import DSGRN

import numpy as np
import json


net_spec = 'X0 : (~X1)(~X3)\nX1 : (~X2)(~X4)\n X2 : (X0)(~X5)\nX3 : (~X4)\nX4 : (X2)(~X1)(~X3)\nX5 : (~X2)(~X5)'

network = DSGRN.Network(net_spec)
parameter_graph = DSGRN.ParameterGraph(network)

print('Number of parameters:', parameter_graph.size())

# Get a parameter
par_index = 12

parameter = parameter_graph.parameter(par_index)

# Sample parameter values
sampler = DSGRN.ParameterSampler(network)
par_sample = sampler.sample(parameter)

# Define L, U, and T from sample
D = network.size()
L = np.zeros([D, D])
U = np.zeros([D, D])
T = np.zeros([D, D])

# Get a dictionary from sample
sample_dict = json.loads(par_sample)

# Get values of L, U, and T from dictionary
for key, value in sample_dict['Parameter'].items():
    # Get parameter (L, U, or T)
    par_type = key[0]
    # Extract variable names
    node_names = [name.strip() for name in key[2:-1].split('->')]
    node_indices = [network.index(node) for node in node_names]
    if par_type == 'L':
        L[tuple(node_indices)] = value
    elif par_type == 'U':
        U[tuple(node_indices)] = value
    else: # T
        T[tuple(node_indices)] = value

L_new = 0.75*np.array([[0,0,3,0,0,0],[1,0,0,0,5,0],[0,2,0,0,5,7],[1,0,0,0,5,0],[0,2,0,4,0,0],[0,0,3,0,0,6]])
U_new = 2*np.array([[0,0,3,0,0,0],[1,0,0,0,5,0],[0,2,0,0,5,7],[1,0,0,0,5,0],[0,2,0,4,0,0],[0,0,3,0,0,6]])
T_new = 3*np.array([[0,0,3,0,0,0],[1,0,0,0,5,0],[0,2,0,0,5,7],[1,0,0,0,5,0],[0,2,0,4,0,0],[0,0,3,0,0,6]])

# Get parameter index from sample
sample_par_index = DSGRN.par_index_from_sample(parameter_graph, L, U, T)
example_par_index =DSGRN.par_index_from_sample(parameter_graph, L_new, U, T)
example_par_index =DSGRN.par_index_from_sample(parameter_graph, L, U_new, T)
example_par_index =DSGRN.par_index_from_sample(parameter_graph, L, U_new, T_new)
sample_par_index_new = DSGRN.par_index_from_sample(parameter_graph, L_new, U_new, T_new)

if not (sample_par_index == par_index):
    print('Wrong parameter node!', par_index, sample_par_index)
else:
    print('Parameter node found correctly!')