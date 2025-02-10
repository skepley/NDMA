import json

import DSGRN
import numpy as np

from ndma.model.model import Model
from ndma.model.restricted_model import RestrictedHillModel
from ndma.DSGRNintegration import DSGRN_functionalities


net_spec = """x : x + y
            y : y (~x)"""

network = DSGRN.Network(net_spec)
parameter_graph = DSGRN.ParameterGraph(network)

par_index = 0 # DSGRN region of interest
parameter = parameter_graph.parameter(par_index)

A = Model.Model_from_string(net_spec)
A_restricted = RestrictedHillModel.Model_from_Model(A)
print('Full model :\n', A)
print('Restricted model :\n', A_restricted)

# find parameters
pars, indices_sources, indices_targets = DSGRN_functionalities.from_region_to_deterministic_point(network, par_index)
hill = 3.2
print('Evaluating the restricted model at the given parameters')
x = np.array([2.3,5.4])
print(A_restricted(x, hill, pars))